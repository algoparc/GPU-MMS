/*
 * (C) Copyright 2016-2018 Ben Karsin, Nodari Sitchinava
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */


#include<stdio.h>
#include<iostream>
#include<fstream>
#include<vector>
#include<cmath>
#include<random>
#include<algorithm>
#include "params.h"
#include "cmp.hxx"
#include "basecase/squareSort.hxx"

// Print out current contents of heap.  Used for debugging
__device__ void printHeap(int* heap) {
  if(threadIdx.x==0) {
    for(int i=1; i<=K; i*=2) {
      for(int j=0; j<i; j++) {
        for(int k=0; k<B; k++) {
          printf("%d ", heap[(i-1+j)*B+k]);
        }
        printf("- ");
      }
      printf("\n");
    }
printf("\n\n");
  }
}

/* Merge two nodes of our minBlockHeap.  Each node contains 32 elements and the merge
is done with 32 threads (1 warp) using a bitonic merge network with shfl */
template<typename T, fptr_t f>
__forceinline__ __device__ bool xorMergeNodes(T* a, T* b, T* out) {
  int tid = threadIdx.x%W;
  bool direction=true;
  T aVal = a[tid];
  T bVal = b[B-1-tid];
  T aTemp;
  T bTemp;
  bool down;
  bool testA;
  bool testB;

  aTemp = myMin<T,f>(aVal,bVal); 
  bVal = myMax<T,f>(aVal,bVal);
  aVal = aTemp;

  direction = f(b[B-1], a[B-1]);

#pragma unroll
  for(int i=B/2; i>0; i=(i>>1)) {
    down = !(tid & i);

    aTemp = shfl_wrapper(aVal, i, W);
    bTemp = shfl_wrapper(bVal, i, W);

    testA = f(aVal, aTemp);
    testB = f(bVal, bTemp);
    testA = testA ^ down;
    testB = testB ^ down;
    if(testA) aVal = aTemp;
    if(testB) bVal = bTemp;
  }
  out[tid] = aVal;

  if(direction) {
    a[tid] = bVal;
    b[tid] = MAXVAL;
  }
  else {
    b[tid] = bVal;
    a[tid] = MAXVAL;
  }
  return direction;
}


// merge two nodes and store the resulting values in registers
template<typename T>
__forceinline__ __device__ T mergeIntoReg(T* heap, int minNode, int maxNode) {
  int tid = threadIdx.x%W;
  T* a = heap+(B*minNode);
  T* b = heap+(B*maxNode);
  T aVal = a[tid];
  T bVal = b[B-1-tid];
  T aTemp;
  T bTemp;
  bool down;

  aTemp = myMin<T>(aVal,bVal); 
  bVal = myMax<T>(aVal,bVal);
  aVal = aTemp;

#pragma unroll
  for(int i=B/2; i>0; i=(i>>1)) {
    down = (tid < (tid ^ ((i<<1)-1)));

    aTemp = shfl_wrapper(aVal, i, W);
    bTemp = shfl_wrapper(bVal, i, W);

//    aTemp = __shfl_xor(aVal, i, W);
//    bTemp = __shfl_xor(bVal, i, W);

    if(down) {
      aVal = myMin<T>(aVal,aTemp);
      bVal = myMin<T>(bVal,bTemp);
    }
    else {
      aVal = myMax<T>(aVal,aTemp);
      bVal = myMax<T>(bVal,bTemp);
    }
  }

  b[tid] = bVal;
  return aVal;
}

// Method to read a block of B elements from an input list and storing it in the leaf of our heap
template<typename T>
__forceinline__ __device__ void fillEmptyLeaf(T* input, T* heap, int listNum, int* start, int* end, int size, int tid) {
  heap[((K-1+listNum)*B)+tid] = MAXVAL;
  if(start[listNum]+tid < end[listNum]) {
    heap[((K-1+listNum)*B)+tid] = input[start[listNum]+(size*listNum)+tid];
    if(tid==0)
      start[listNum] += B;
  }
}

// Each thread searches down path to find node to work on
template<typename T>
__forceinline__ __device__ int findNodeInPath(T* heap, int nodeGroup, int LOGK) {
  int nodeIdx=0;
  for(int i=0; i<nodeGroup; i++) {
    nodeIdx ++;
    nodeIdx += (cmp(heap[(2*nodeIdx+1)*B+(B-1)], heap[(2*nodeIdx+2)*B+(B-1)]));
  }
  return nodeIdx;
}

template<typename T>
__forceinline__ __device__ int fillRegsFromPathBit(T* elts, T* heap, int tid) {
  int path=1;
#pragma unroll
  for(int i=0; i<PL<<1; i+=2) {
    path = path << 1;
    elts[i] = heap[(path-1)*B+tid];
    elts[i+1] = heap[(path)*B+tid];
    path += (heap[(path-1)*B+(B-1)] > heap[(path)*B+(B-1)]);
  }
  return path;
}

// Each thread searches down path to find node to work on
template<typename T, fptr_t f>
__forceinline__ __device__ void fillRegsFromPath(T* elts, T* heap, int* path, int tid) {
  int idx=0;
  path[0]=0;

#pragma unroll
  for(int i=0; i<PL; i++) {
    elts[idx] = heap[((2*path[i]+1)<<LOGB)+tid];
    elts[idx+1] = heap[((2*path[i]+2)<<LOGB)+(B-1)-tid];
    idx+=2;

    path[i+1] = path[i]*2+1;
    path[i+1] += !f(heap[(path[i+1]<<LOGB)+(B-1)], heap[((path[i+1]+1)<<LOGB)+(B-1)]);
  }
}

template<typename T, fptr_t f>
__forceinline__ __device__ void xorMergeGeneral(T* elts, T* heap, int tid) {

  T temp[2*PL];
  bool down;

#pragma unroll
  for(int j=0; j<PL<<1; j+=2) {
      temp[j]=myMin<T,f>(elts[j],elts[j+1]);
      elts[j+1] = myMax<T,f>(elts[j],elts[j+1]);
      elts[j]=temp[j];
  }

#pragma unroll
  for(int i=B/2; i>0; i=(i>>1)) {

#pragma unroll
    for(int j=0; j<PL<<1; j++) {
      temp[j] = shfl_wrapper(elts[j], i, W);
    }
    down = (tid & i);


#pragma unroll
    for(int j=0; j<PL<<1; j++) {
      if(down) {
        elts[j] = myMax<T,f>(temp[j],elts[j]);
      }
      else {
        elts[j] = myMin<T,f>(temp[j],elts[j]);
      }
    }
    __syncwarp();
  }     
}

// Fill an empty node with the merge of its children
// Pipelined version to increase ILP
template<typename T, fptr_t f>
__device__ int heapifyEmptyNodePipeline(T* heap, int* path, int tid) {
  T elts[2*PL];
  fillRegsFromPath<T,f>(elts, heap, path, tid);

  xorMergeGeneral<T,f>(elts, heap, tid);

  int idx=0;
#pragma unroll
  for(int i=0; i<PL; i++) {
    heap[(path[i]<<5)+tid] = elts[idx];
    heap[((path[i+1]-1+((path[i+1]&1)<<1))<<5)+tid] = elts[idx+1];
    idx+=2;
  }
  return path[PL];

}

// Fill an empty node with the merge of its children
template<typename T, fptr_t f>
__device__ int heapifyEmptyNode(T* node, int nodeIdx, int altitude, int tid) {
  bool direction;
  for(int i=0; i<altitude; i++) {
    direction = xorMergeNodes<T,f>(node+((2*nodeIdx+1)*B), node+((2*nodeIdx+2)*B), node+(nodeIdx*B));
    nodeIdx+=(nodeIdx+1);
    if(direction) //If new empty is right child
      nodeIdx++;
  }
  // Return index of leaf node that needs to be filled from global memory read
  return nodeIdx; 
}

// Build a minBlockHeap bottom-up by merging pairs of nodes and filling empty nodes from global memory
template<typename T, fptr_t f>
__device__ void buildHeap(T* input, T* heap, int* start, int* end, int size, int tid) {
  int nodeIdx;
  // Fill leaf nodes
#pragma unroll
  for(int i=0; i<K; i++) {
    if(start[i]+tid < end[i])
      heap[((K-1+i)<<LOGB)+tid] = input[start[i]+(size*i)+tid]; 
    else
      heap[((K-1+i)<<LOGB)+tid] = MAXVAL;
  }
  // May need a warp synchronization after adding B to start[tid]
  if(tid < K)
    start[tid] += B;
  __syncwarp();

// Go through each level of the tree, merging children to make new nodes
// each merge propagates down to leaf where a new block is taken from input
  int nodesAtLevel=K>>1;
#pragma unroll
  for(int i=1; nodesAtLevel > 0; i++) {
#pragma unroll
    for(int j=0; j<nodesAtLevel; j++) {
      nodeIdx = (nodesAtLevel-1+j);
      nodeIdx = heapifyEmptyNode<T,f>(heap, nodeIdx, i, tid);
      fillEmptyLeaf<T>(input, heap, nodeIdx-(K-1), start, end, size, tid);
      __syncwarp();
    }
    nodesAtLevel = nodesAtLevel >> 1;
  }
}

// Merge K lists into one using 1 warp
template<typename T, fptr_t f>
__device__ void multimergePipeline(T* input, T* output, int* start, int* end, int size, int outputOffset) {
  int warpInBlock = threadIdx.x>>5;
  int tid = threadIdx.x&(W-1);
  __shared__ T heapData[B*(2*K-1)*(THREADS>>5)]; // Each warp in the block needs its own shared memory
  T* heap = heapData+((B*(2*K-1))*warpInBlock);


  int path[PL+1];


  buildHeap<T,f>(input, heap, start, end, size,tid);
  __syncwarp();

  int outputIdx=tid+outputOffset;
  int nodeIdx;

  while(heap[B-1] != MAXVAL) {
    output[outputIdx] = heap[tid];
    outputIdx += B;
    nodeIdx = heapifyEmptyNodePipeline<T,f>(heap, path, tid);
    fillEmptyLeaf<T>(input, heap, nodeIdx-(K-1), start, end, size, tid);
    __syncwarp();
  }

  __syncwarp();

  if(heap[tid] != MAXVAL) {
    output[outputIdx] = heap[tid];
  }

}


template<typename T, fptr_t f>
__device__ void mmp(T* input, T* output, int* start, int* end, int size, int outputOffset) {
  int warpInBlock = threadIdx.x>>5;
  int tid = threadIdx.x&(W-1);
  __shared__ T heapData[B*(2*K-1)*(THREADS>>5)]; // Each warp in the block needs its own shared memory
  T* heap = heapData+((B*(2*K-1))*warpInBlock);

  int path[PL+1];

  buildHeap<T,f>(input, heap, start, end, size,tid);
    __syncwarp(); // unnecessary

  int outputIdx=tid+outputOffset;
  int nodeIdx;

  while(heap[B-1] != MAXVAL) {
    output[outputIdx] = heap[tid];
    outputIdx += B;
    nodeIdx = heapifyEmptyNodePipeline<T,f>(heap, path, tid);
    fillEmptyLeaf<T>(input, heap, nodeIdx-(K-1), start, end, size, tid);
  }

  __syncwarp();

  if(heap[tid] != MAXVAL) {
    output[outputIdx] = heap[tid];
  }
}

template<typename T, fptr_t f>
__device__ void multimergePipelineEdgeCaseFull(T* input, T* output, int* start, int* end, int size, int outputOffset) {
  int warpInBlock = threadIdx.x>>5;
  int tid = threadIdx.x&(W-1);
  __shared__ T heapData[B*(2*K-1)*(THREADS>>5)]; // Each warp in the block needs its own shared memory
  T* heap = heapData+((B*(2*K-1))*warpInBlock);

  int path[PL+1];

  buildHeap<T,f>(input, heap, start, end, size,tid);

  int outputIdx=tid+outputOffset;
  int nodeIdx;

  while(heap[B-1] != MAXVAL) {
    output[outputIdx] = heap[tid];
    outputIdx += B;
    nodeIdx = heapifyEmptyNodePipeline<T,f>(heap, path, tid);
    fillEmptyLeaf<T>(input, heap, nodeIdx-(K-1), start, end, size, tid);
  }
  // Write the last remaining node to global memory
  if(heap[tid] != MAXVAL) {
    output[outputIdx] = heap[tid];
  }
}

template<typename T, fptr_t f>
__device__ void multimergePipelineEdgeCasePartial(T* input, T* output, int* start, int* end, int size, int smallerSize, int outputOffset, int L) {
  
  
  int tid = threadIdx.x&(W-1);
  __shared__ T heap[B*(2*K-1)];

  int path[PL+1];

  buildHeap<T,f>(input, heap, start, end, size,tid);

  /*
  int outputIdx=tid+outputOffset;
  int nodeIdx;

  while(heap[B-1] != MAXVAL) {
    output[outputIdx] = heap[tid];
    outputIdx += B;
    nodeIdx = heapifyEmptyNodePipeline<T,f>(heap, path, tid);
    fillEmptyLeaf<T>(input, heap, nodeIdx-(K-1), start, end, size, tid);
  }

  // If there is still something in the highest node of the heap, write it out to global memory
  // TODO: Fix this
  if(heap[tid] != MAXVAL) {
    output[outputIdx] = heap[tid];
  }
  */

}
// Merge K lists into one using 1 warp
template<typename T, fptr_t f>
__device__ void multimerge(T* input, T* output, int* start, int* end, int size, int outputOffset) {
  int warpInBlock = threadIdx.x>>5;
  int tid = threadIdx.x&(W-1);
  __shared__ T heapData[B*(2*K-1)*(THREADS/W)]; // Each warp in the block needs its own shared memory
  T* heap = heapData+((B*(2*K-1))*warpInBlock);

  int LOGK=PL;

  buildHeap<T,f>(input, heap, start, end, size, tid);
  int outputIdx=tid+outputOffset;
  int nodeIdx;

  while(heap[B-1] != MAXVAL) {
    output[outputIdx] = heap[tid];
    outputIdx += B;
    nodeIdx = heapifyEmptyNode<T,f>(heap, 0, LOGK, tid);
    fillEmptyLeaf<T>(input, heap, nodeIdx-(K-1), start, end, size, tid);
  }

  if(heap[tid] != MAXVAL) {
    output[outputIdx] = heap[tid];
  }
}

template<typename T>
__device__ void blockBuildHeap(T* input, T* heap, int* start, int* end, int size) {
  
}

template<typename T>
__device__ int findListToRead(T* heap, int LOGK) {
  int nodeIdx=0;
  for(int i=0; i<LOGK-1; i++) {
    nodeIdx = 2*nodeIdx + 1;
    nodeIdx += (heap[nodeIdx*B + (B-1)] > heap[(nodeIdx+1)*B + (B-1)]);
  }
  return nodeIdx - (K-1);
}

template<typename T>
__forceinline__ __device__ int findMergeNode(T* heap, int warpInBlock, bool* dir) {
  int nodeIdx=0;
  bool tempDir;
  for(int i=0; i<warpInBlock; i++) {
    nodeIdx = 2*nodeIdx + 1;
    tempDir = (heap[nodeIdx*B + (B-1)] > heap[(nodeIdx+1)*B + (B-1)]);
    nodeIdx += tempDir;
  }
  *dir = !tempDir;
  return nodeIdx;
}

template<typename T>
__device__ void blockMultimerge(T* input, T* output, int* start, int* end, int size, int outputOffset) {
  int warpInBlock = threadIdx.x/W;
  int tid = threadIdx.x%W;
  __shared__ T heap[B*(2*K-1)]; // Each block shares a single heap

  int LOGK=1;
  int tempK=K;
  while(tempK >>= 1) LOGK++;

  if(warpInBlock == 0) {
    buildHeap<T>(input, heap, start, end, size, tid);
  }
  int outputIdx = tid+outputOffset;
__syncthreads();
  if(warpInBlock==0) {
int count=0;
    while(heap[tid] != MAXVAL) {
    // write out root
      output[outputIdx] = heap[tid];
      outputIdx+=W;
count++;
      __syncthreads();
      __syncthreads();
    }
  }
  else if(warpInBlock == LOGK){
    int listNum;
    T readVal;
    while(heap[0]!=MAXVAL) {
      readVal = MAXVAL;
      listNum = findListToRead<T>(heap, LOGK); // search down to leaf
    // read in new leaf to register
      if(start[listNum]+tid < end[listNum]) {
        readVal = input[start[listNum]+(size*listNum)+tid];
      } 
        start[listNum] += B;
    __syncthreads();

    heap[(K-1+listNum)*B+tid] = readVal;   // write register to heap node
    __syncthreads();
    }
  }
  else {
    int minNode;
    int maxNode;
    T regVal;
    bool dir;
    while(heap[0] != MAXVAL) {
      // search down path
      minNode = findMergeNode<T>(heap, warpInBlock, &dir); 
      maxNode = minNode-1 + dir + dir;
    // merge nodes & write max 
      regVal = mergeIntoReg<T>(heap, minNode, maxNode);
      __syncthreads();
      heap[((maxNode-1)/2)*B + tid] = regVal; // write register to heap node
      __syncthreads();
    }
  }
}
