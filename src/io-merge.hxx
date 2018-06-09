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


/* This file is depricated.  The new io-merge-gen.hxx file is now used because it is generalized to
work with key/value pairs and other datatypes.*/

#include<stdio.h>
#include<iostream>
#include<fstream>
#include<vector>
#include<cmath>
#include<random>
#include<algorithm>
#include "params.h"
#include "cmp.hxx"

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
template<typename T>
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

  aTemp = min(aVal,bVal); 
  bVal = max(aVal,bVal);
  aVal = aTemp;

  direction = (b[B-1] < a[B-1]);

#pragma unroll
  for(int i=B/2; i>0; i=(i>>1)) {
//    down = (tid < (tid ^ ((i<<1)-1)));
    down = !(tid & i);

    aTemp = __shfl_xor(aVal, i, W);
    bTemp = __shfl_xor(bVal, i, W);

    testA = (aVal < aTemp);
    testB = (bVal < bTemp);
    testA = testA ^ down;
    testB = testB ^ down;
    if(testA) aVal = aTemp;
    if(testB) bVal = bTemp;
/*
    if((aVal < aTemp) ^ down) 
      aVal = aTemp;
    if((bVal < bTemp) ^ down) 
      bVal = bTemp;
*/
   /* 
   if(down) {
      aVal = min(aVal,aTemp);
      bVal = min(bVal,bTemp);
    }
    else {
      aVal = max(aVal,aTemp);
      bVal = max(bVal,bTemp);
    }
*/
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

  aTemp = min(aVal,bVal); 
  bVal = max(aVal,bVal);
  aVal = aTemp;

#pragma unroll
  for(int i=B/2; i>0; i=(i>>1)) {
    down = (tid < (tid ^ ((i<<1)-1)));

    aTemp = __shfl_xor(aVal, i, W);
    bTemp = __shfl_xor(bVal, i, W);

    if(down) {
      aVal = min(aVal,aTemp);
      bVal = min(bVal,bTemp);
    }
    else {
      aVal = max(aVal,aTemp);
      bVal = max(bVal,bTemp);
    }
  }

  b[tid] = bVal;
  return aVal;
}

// Method to read a block of B elements from an input list and storing it in the leaf of our heap
template<typename T>
__forceinline__ __device__ void fillEmptyLeaf(T* input, T* heap, int listNum, int* start, int* end, int size, int tid) {
//  int tid = threadIdx.x%W;
//if(tid==0)printf("filling listNum:%d, idx:%d\n", listNum, (K-1+listNum)*B+tid);
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
    nodeIdx += (heap[(2*nodeIdx+1)*B+(B-1)] < heap[(2*nodeIdx+2)*B+(B-1)]);
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
template<typename T>
__forceinline__ __device__ void fillRegsFromPath(T* elts, T* heap, int* path, int tid) {
  int idx=0;
//  __shared__ int path[PL+1];
  path[0]=0;

#pragma unroll
  for(int i=0; i<PL; i++) {
    elts[idx] = heap[((2*path[i]+1)<<LOGB)+tid];
    elts[idx+1] = heap[((2*path[i]+2)<<LOGB)+(B-1)-tid];
    idx+=2;

    path[i+1] = path[i]*2+1;
    path[i+1] += (heap[(path[i+1]<<LOGB)+(B-1)] > heap[((path[i+1]+1)<<LOGB)+(B-1)]);
//    if(threadIdx.x==0) printf("node:%d:%d(pos:%d), node:%d:%d(pos:%d), path:%d\n", path[i]*2+1, heap[path[i+1]*B+(B-1)], path[i+1]*B+(B-1), path[i]*2+2, heap[(path[i+1]+1)*B+(B-1)], (path[i+1]+1)*B+(B-1), path[i+1]);
  }
//  return path;
}

template<typename T>
__forceinline__ __device__ void xorMergeGeneral(T* elts, T* heap, int tid) {

  T temp[2*PL];
  bool down;
//  bool test[2*PL];
//  if(threadIdx.x==0) {
//    printf("thread:%d, elts:[%d,%d,%d,%d]\n", threadIdx.x, elts[0], elts[1],elts[2],elts[3]);
//  printf("\n");
//  }
#pragma unroll
  for(int j=0; j<PL<<1; j+=2) {
    if(elts[j] > elts[j+1]) {
      temp[j] = elts[j];
      elts[j] = elts[j+1];
      elts[j+1] = temp[j];
//      temp[j]=min(elts[j],elts[j+1]);
//      elts[j+1] = max(elts[j],elts[j+1]);
//      elts[j]=temp[j];
    }
  }

#pragma unroll
  for(int i=B/2; i>0; i=(i>>1)) {

#pragma unroll
    for(int j=0; j<PL<<1; j++) {
      temp[j] = __shfl_xor(elts[j], i, W);
    }
    down = (tid & i);
//__syncthreads();

/*
#pragma unroll
    for(int j=0; j<PL<<1; j++) {
      test[j] = (elts[j] > temp[j]);
      test[j] = test[j] ^ (!down);
      test[j]--;
    }
#pragma unroll
    for(int j=0; j<PL<<1; j++) {
//      if(test[j]) elts[j] = temp[j];
      elts[j] += (temp[j]-elts[j])&(test[j]);
//      elts[j] = elts[j]*(!test[j]) + temp[j]*test[j];
    }
*/
/*
#pragma unroll
  for(int j=0; j<PL<<1; j++) {
    if(down != (elts[j] < temp[j]))
      elts[j] = temp[j];
  }
*/

#pragma unroll
    for(int j=0; j<PL<<1; j++) {
      if(down) {
        elts[j] = max(temp[j],elts[j]);
      }
      else {
        elts[j] = min(temp[j],elts[j]);
      }
    }
  }     
//    printf("thread:%d, elts:[%d,%d,%d,%d]\n", threadIdx.x, elts[0], elts[1],elts[2],elts[3]);
}

// Fill an empty node with the merge of its children
template<typename T>
//__device__ int heapifyEmptyNodePipeline(T* heap, int* path, int tid) {
__device__ int heapifyEmptyNodePipeline(T* heap, int* path, int tid) {
//  int nodeIdx = findNodeInPath(heap, nodeGroup, LOGK);
  T elts[2*PL];
//  int* path;
  fillRegsFromPath(elts, heap, path, tid);
//  int path = fillRegsFromPathBit(elts, heap, tid);
//  int pathTemp = path;

  xorMergeGeneral<T>(elts, heap, tid);
/*
  int idx=2*PL-1;
#pragma unroll
  for(int i=0; i<PL; i++) {
    heap[((pathTemp^1)-1)*B+tid] = elts[idx];
    pathTemp = pathTemp >> 1;
    heap[(pathTemp-1)*B+tid] = elts[idx-1];
    idx-=2;
  }
  return path-1;
*/

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
template<typename T>
__device__ int heapifyEmptyNode(T* node, int nodeIdx, int altitude, int tid) {
  bool direction;
  for(int i=0; i<altitude; i++) {
    direction = xorMergeNodes<T>(node+((2*nodeIdx+1)*B), node+((2*nodeIdx+2)*B), node+(nodeIdx*B));
    nodeIdx+=(nodeIdx+1);
    if(direction) //If new empty is right child
      nodeIdx++;
  }
  // Return index of leaf node that needs to be filled from global memory read
  return nodeIdx; 
}

// Build a minBlockHeap bottom-up by merging pairs of nodes and filling empty nodes from global memory
template<typename T>
__device__ void buildHeap(T* input, T* heap, int* start, int* end, int size, int tid) {
//  int tid = threadIdx.x%W;
  int nodeIdx;
  // Fill leaf nodes
#pragma unroll
  for(int i=0; i<K; i++) {
    if(start[i]+tid < end[i])
      heap[((K-1+i)<<LOGB)+tid] = input[start[i]+(size*i)+tid]; 
    else
      heap[((K-1+i)<<LOGB)+tid] = MAXVAL;
  }
  if(tid < K)
    start[tid] += B;

// Go through each level of the tree, merging children to make new nodes
// each merge propagates down to leaf where a new block is taken from input
  int nodesAtLevel=K>>1;
#pragma unroll
  for(int i=1; nodesAtLevel > 0; i++) {
#pragma unroll
    for(int j=0; j<nodesAtLevel; j++) {
      nodeIdx = (nodesAtLevel-1+j);
      nodeIdx = heapifyEmptyNode<T>(heap, nodeIdx, i, tid);
      fillEmptyLeaf<T>(input, heap, nodeIdx-(K-1), start, end, size, tid);
    }
    nodesAtLevel = nodesAtLevel >> 1;
  }
}

// Merge K lists into one using 1 warp
template<typename T>
__device__ void multimergePipeline(T* input, T* output, int* start, int* end, int size, int outputOffset) {
//  int warpInBlock = threadIdx.x/W;
//  int tid = threadIdx.x%W;
//if(threadIdx.x==0 && blockIdx.x==0)printf("%d\n", size);
  int warpInBlock = threadIdx.x>>5;
  int tid = threadIdx.x&(W-1);
  __shared__ T heapData[B*(2*K-1)*(THREADS>>5)]; // Each warp in the block needs its own shared memory
  T* heap = heapData+((B*(2*K-1))*warpInBlock);

//  __shared__ int pathData[(PL+1)*(THREADS/W)];
//  int* path = pathData+((PL+1)*warpInBlock);

  int path[PL+1];

//  int LOGK=0;
//  int tempK=K;
//  while(tempK >>= 1) LOGK++;
//  LOGK = CONSTLOGK;

//  int threadsPerNode = W/PL;
//  int tidForNode = tid%threadsPerNode;
//  int nodeGroup = tid/threadsPerNode;

  buildHeap<T>(input, heap, start, end, size,tid);
//for(int i=0; i<THREADS/W; i++) {
//}

  int outputIdx=tid+outputOffset;
  int nodeIdx;

  while(heap[B-1] != MAXVAL) {
    output[outputIdx] = heap[tid];
    outputIdx += B;
  //  nodeIdx = heapifyEmptyNode<T>(heap, 0, LOGK, tid);
    nodeIdx = heapifyEmptyNodePipeline<T>(heap, path, tid);
//    nodeIdx = heapifyEmptyNodePipeline<T>(heap, tid);
    fillEmptyLeaf<T>(input, (T*)heap, nodeIdx-(K-1), start, end, size, tid);
  }

  if(heap[tid] != MAXVAL) {
    output[outputIdx] = heap[tid];
  }

}

// Merge K lists into one using 1 warp
template<typename T>
__device__ void multimerge(T* input, T* output, int* start, int* end, int size, int outputOffset) {
  int warpInBlock = threadIdx.x>>5;
  int tid = threadIdx.x&(W-1);
  __shared__ T heapData[B*(2*K-1)*(THREADS/W)]; // Each warp in the block needs its own shared memory
  T* heap = heapData+((B*(2*K-1))*warpInBlock);

//if(tid==0) {
//  printf("warp:%d, heapOffset:%d, outputOffset:%d, LOGK:%d\n", warpInBlock, ((B*(2*K-1))*warpInBlock), outputOffset, PL);
//}

//  int LOGK=0;
//  int tempK=K;
//  while(tempK >>= 1) LOGK++;
//if(threadIdx.x==0 && blockIdx.x==0)printf("LOGK:%d\n", LOGK);
  int LOGK=PL;

  buildHeap<T>(input, heap, start, end, size, tid);
  int outputIdx=tid+outputOffset;
  int nodeIdx;

  while(heap[B-1] != MAXVAL) {
    output[outputIdx] = heap[tid];
    outputIdx += B;
    nodeIdx = heapifyEmptyNode<T>(heap, 0, LOGK, tid);
    fillEmptyLeaf<T>(input, (T*)heap, nodeIdx-(K-1), start, end, size, tid);
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
//    printf("node:%d, lIdx:%d, lVal:%d, rIdx:%d, rVal:%d, right?:%d\n", nodeIdx, nodeIdx*B + (B-1), heap[nodeIdx*B + (B-1)], (nodeIdx+1)*B + (B-1),heap[(nodeIdx+1)*B + (B-1)], (heap[nodeIdx*B + (B-1)] > heap[(nodeIdx+1)*B + (B-1)]));
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
//printf("node:%d, lVal:%d, lPos:%d, rVal:%d, rPos:%d, +1?:%d\n", nodeIdx, heap[nodeIdx*B + (B-1)], nodeIdx*B+(B-1), heap[(nodeIdx+1)*B + (B-1)], (nodeIdx+1)*B + (B-1), tempDir);
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

//  blockBuildHeap<T>(input, heap, start, end, size);
  if(warpInBlock == 0) {
    buildHeap<T>(input, heap, start, end, size, tid);
  }
  int outputIdx = tid+outputOffset;
  //int nodeIdx;
//  __syncthreads();
//  if(threadIdx.x==0 && blockIdx.x==0) {
//if(blockIdx.x==0) {
//    printHeap(heap);
//}
//  }
__syncthreads();
  if(warpInBlock==0) {
int count=0;
    while(heap[tid] != MAXVAL) {
    // write out root
      output[outputIdx] = heap[tid];
      outputIdx+=W;
count++;
//printf("outputOffset:%d, outputIdx:%d\n", outputOffset, outputIdx);
      __syncthreads();
      __syncthreads();
    }
  }
  else if(warpInBlock == LOGK){
    int listNum;
    T readVal;
    while(heap[0]!=MAXVAL) {
//    while(!done) {
      readVal = MAXVAL;
      listNum = findListToRead<T>(heap, LOGK); // search down to leaf
    // read in new leaf to register
      if(start[listNum]+tid < end[listNum]) {
        readVal = input[start[listNum]+(size*listNum)+tid];
      } 
//printf("listNum:%d, val:%d\n", listNum, readVal);
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
//if(warpInBlock==3)
//printf("tid:%d, minNode:%d, min[tid]:%d, maxNode:%d, max[tid]:%d\n", tid, minNode, heap[minNode*B+tid],maxNode,heap[maxNode*B+tid]);
      regVal = mergeIntoReg<T>(heap, minNode, maxNode);
      __syncthreads();
//printf("warp:%d, minNode:%d, maxNode:%d, tid:%d, regVal:%d\n", warpInBlock, minNode, maxNode, tid, regVal);
      heap[((maxNode-1)/2)*B + tid] = regVal; // write register to heap node
      __syncthreads();
    }
  }
}
