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

// #define DEBUG

#include<stdio.h>
#include<iostream>
#include<fstream>
#include<vector>
#include<cmath>
#include<random>
#include<algorithm>
#include "params.h"
#include "io-merge-gen.hxx"
#include "warp-partition.hxx"
#include "pad.hxx"

// Function that does a single level of multiway mergesort
template<typename T, fptr_t f>
__global__ void multimergeLevel(T* data, T* output, int* pivots, long size, int tasks);

template<typename T>
__global__ void copy(T* arr1, T* arr2, long offset, long N);

template <typename T, fptr_t f>
__global__ void testSortedSegments(int* d_arr, int segmentSize, int N);

// Depricated function, was an attempt to implement another optimization that didn't turn out to be worthwhile.
//template<typename T>
//__global__ void blockMultimergeLevel(T* data, T* output, int* pivots, int size, int tasks, int P);

// Functions used for debugging and performance analysis
__global__ void init_count() {
  if(threadIdx.x==0 && blockIdx.x==0) tot_cmp=0;
}
__global__ void print_count() {
  if(threadIdx.x==0 && blockIdx.x==0) printf("cmps:%d\n", tot_cmp);
}


/* Main CPU function that sorts an input and writes the result to output 
   Parameters:
     T* input:    The DEVICE array input to merge
     T* output:   The DEVICE array output
     T* h_data:   The same array values as input, but exists on the host
     P:           An integer representing the number of blocks to launch in our grid
     N:           The size of the array to merge
   Notes:
     For the base case, we use an integer number of blocks that is different from P.
     P is the number of blocks to use for every kernel launch except our base case.
*/
template<typename T, fptr_t f>
T* multimergesort(T* input, T* output, T* h_data, int P, int N) {
  int* pivots;
  int pivotsMemorySize = sizeof(int) * K * (1 + (N+M*K-1)/M/K);
  cudaError_t err;
  cudaMalloc((void**) &pivots, pivotsMemorySize);
  int tasks;
  int edgeCaseTaskSize;
  T* list[2];
  list[0]=input;
  list[1]=output;
  bool listBit = false;
  int baseBlocks=((N/M)/(THREADS_BASE_CASE/W));

// Sort the base case into blocks of 1024 elements each
  squareSort<T,f><<<baseBlocks,THREADS_BASE_CASE>>>(input, N);
  cudaDeviceSynchronize();

/*
  Perform successive merge rounds.
  We are sorting lists with size listSize, which starts at M=1024,
  because we just finished sorting the base cases of contiguous
  1024 elements.
*/
  for(int listSize=M; listSize < N; listSize *= K) {

    tasks = N/listSize/K + ((N%(K*listSize))>0);
    edgeCaseTaskSize = (N%(K*listSize)>0) ? N%(K*listSize) : K*listSize;
    // printf("tasks: %d \n  edgeCaseTaskSize: %d\nlistSize %d\n", tasks, edgeCaseTaskSize, listSize);

    if(tasks > P) { // If each block has its own designated task all to itself
      findPartitions<T,f><<<tasks,THREADS>>>(list[listBit], pivots, listSize, tasks, edgeCaseTaskSize);
      cudaDeviceSynchronize();
      #ifdef DEBUG
      err = cudaGetLastError();
      if (err != cudaSuccess) {
        printf("Many tasks, find partitions: %s\n", cudaGetErrorString(err));
      }
      #endif
      multimergeLevel<T,f><<<tasks,THREADS>>>(list[listBit], list[!listBit], pivots, listSize, tasks);
      #ifdef DEBUG
      cudaDeviceSynchronize();
      err = cudaGetLastError();
      if (err != cudaSuccess) {
        printf("Many tasks, merge: %s\n", cudaGetErrorString(err));
      }
      #endif
    }
    else {
      // Each block only does one task
      findPartitions<T,f><<<P,THREADS>>>(list[listBit], pivots, listSize, tasks, edgeCaseTaskSize);
      cudaDeviceSynchronize();

      #ifdef DEBUG
      printPartitions<<<1,1>>>(pivots, tasks*(P/tasks));
      cudaDeviceSynchronize();
      err = cudaGetLastError();
      if (err != cudaSuccess) {
        printf("Few tasks, find partitions: %s\n", cudaGetErrorString(err));
      }
      #endif
      multimergeLevel<T,f><<<P,THREADS>>>(list[listBit], list[!listBit], pivots, listSize, tasks);
      #ifdef DEBUG
      cudaDeviceSynchronize();
      err = cudaGetLastError();
      if (err != cudaSuccess) {
        printf("Few tasks, merge: %s\n", cudaGetErrorString(err));
      }
      #endif
    }
    listBit = !listBit; // Switch input/output arrays
  }

  cudaDeviceSynchronize();
  cudaFree(pivots);

  return list[listBit]; // This returns the array that was last used as output
}

/*
    Performs one level of merge, assuming that the pivots have already been determined.
    Parameters:

    T* data     -- The input to merge. The data should already pass in any offsets, so the merge will begin at index 0 relative to the data pointer.
    T* output   -- The location to store the output.
    int* pivots -- Where the pivots are stored. There should be K pivots for every active warp, with one additional set of pivots, so the length of relevant pivot array should be K*(activeWarps + 1)
    long size   -- The size of each sorted subarray to be merged. We will merge these sorted subarrays into sizes of K*size
    int tasks   -- The number of tasks. Each task is defined as a single merge of four subarray sections
    int P       -- The number of blocks (equivalent to gridDim.x)
*/

template<typename T, fptr_t f>
__global__ void multimergeLevel(T* data, T* output, int* pivots, long size, int tasks) {
  __shared__ int start[K];
  __shared__ int end[K];

  int blocksPerTask = gridDim.x/tasks;
  if(blocksPerTask == 0) blocksPerTask=1;
  int myTask = blockIdx.x/blocksPerTask;
  long taskOffset = size*K*myTask;

  
  int outputOffset=0;

  if(myTask < tasks) {

    if(threadIdx.x<K) {
      start[threadIdx.x] = pivots[(blockIdx.x*K)+threadIdx.x];
    }

    if(threadIdx.x<K) 
      end[threadIdx.x] = (myTask<tasks-1)*size + (myTask==tasks-1)*pivots[blocksPerTask*tasks*K+threadIdx.x];
    if(blockIdx.x % blocksPerTask < blocksPerTask-1 && threadIdx.x < K)
      end[threadIdx.x] = pivots[((blockIdx.x+1)*K)+threadIdx.x];
    __syncthreads();

    for(int i=0; i<K; i++)
      outputOffset+=start[i];
    __syncthreads();

    multimergePipeline<T,f>(data+taskOffset, output+taskOffset, start, end, size, outputOffset);
  }
}

template<typename T>
__global__ void copy(T* arr1, T* arr2, long offset, long N){
  if (blockIdx.x*THREADS + threadIdx.x + offset < N)
    arr2[blockIdx.x*THREADS + threadIdx.x + offset] = arr1[blockIdx.x*THREADS + threadIdx.x + offset];
}

// Launch with 1 thread and 1 block
template <typename T, fptr_t f>
__global__ void testSortedSegments(int* d_arr, int segmentSize, int N) {
  for (int i = 0; i < N; i += segmentSize) {
    for (int j = 1; j < segmentSize; j++) {
      if (i+j >= N) {
        break;
      }
      if (!f(d_arr[i+j-1], d_arr[i+j])) {
        printf("UNSORTED (WITH SEGMENT SIZES %d) AT INDEX %d AND %d\n", segmentSize, i+j-1, i+j);
        return;
      }
    }
  }
  printf("SORTED WITH SEGMENT SIZES : %d\n", segmentSize);
}

/************************************************************
* Depricated code.  Attempt at another optimization, but did
* not perform as well.
*************************************************************/
/*
template<typename T>
T* blockMultimergesort(T* input, T* output, T* h_data, int P, int N, int (*f)(T,T)) {
  int* pivots;
//printf("numPivots:%d\n", (P+1)*K);
  cudaMalloc(&pivots, (P+1)*K*sizeof(int));
  cudaMemset(&pivots, 0, (P+1)*K*sizeof(int));
  int tasks;
  T* list[2];
  list[0]=input;
  list[1]=output;
  bool listBit = false;
  int baseBlocks=((N/M)/(THREADS/W));

  squareSort<T><<<baseBlocks,THREADS>>>(input,f);
#ifdef DEBUG
  bool correct=true;
  cudaMemcpy(h_data, input, N*sizeof(T), cudaMemcpyDeviceToHost);

  for(int i=0; i<N/M; i++) {
    for(int j=1; j<M; j++) {
      if(h_data[i*M+(j)] < h_data[i*M+(j-1)])
        correct=false;
    }
  }
  if(!correct) printf("base case not sorted!\n");
#endif
  for(int listSize=M; listSize <= (N/K); listSize *= K) {
//  for(int listSize=M; listSize <= M; listSize *= K) {
    tasks = (N/listSize)/K;
    if(tasks > P) {
      for(int i=0; i<tasks/P; i++) {
        findPartitions<T><<<P,W>>>(list[listBit]+(i*P*K*listSize), list[!listBit]+(i*P*K*listSize), pivots, listSize, P*K, P);
        cudaDeviceSynchronize();

#ifdef DEBUG
//  testPartitioning<T><<<P,W>>>(list[listBit]+(i*P*K*listSize), pivots, listSize, P);
#endif

        blockMultimergeLevel<T><<<P,BLOCKTHREADS>>>(list[listBit]+(i*P*K*listSize), list[!listBit]+(i*P*K*listSize), pivots, listSize, P, P);
        cudaDeviceSynchronize();
      }
// Perform remaining tasks
      if(tasks%P > 0) {
        findPartitions<T><<<P,W>>>(list[listBit]+((tasks/P)*P*K*listSize), list[!listBit]+((tasks/P)*P*K*listSize), pivots, listSize, P*K, P);
        cudaDeviceSynchronize();
#ifdef DEBUG
//  testPartitioning<T><<<P,W>>>(list[listBit]+((tasks/P)*P*K*listSize), pivots, listSize, P);
#endif

        blockMultimergeLevel<T><<<P,BLOCKTHREADS>>>(list[listBit]+((tasks/P)*P*K*listSize), list[!listBit]+((tasks/P)*P*K*listSize), pivots, listSize, tasks%P, P);
        cudaDeviceSynchronize();
    
      }
    }
    else {
      findPartitions<T><<<P,W>>>(list[listBit], list[!listBit], pivots, listSize, tasks*K, tasks);
        cudaDeviceSynchronize();
#ifdef DEBUG
//  testPartitioning<T><<<P,W>>>(list[listBit], pivots, listSize, P);
#endif

      blockMultimergeLevel<T><<<P,BLOCKTHREADS>>>(list[listBit], list[!listBit], pivots, listSize, tasks, P);
        cudaDeviceSynchronize();
    }
    listBit = !listBit;
  }
  cudaFree(pivots);
  return list[listBit];
}
*/

/* Main Kernel to merge groups of K lists */
/*
template<typename T>
__global__ void blockMultimergeLevel(T* data, T* output, int* pivots, int size, int tasks, int P) {
//  int totalWarps = P*(THREADS/W);
//  int warpInBlock = threadIdx.x/W;
//  int warpIdx = (blockIdx.x)*(THREADS/W)+warpInBlock;
//  int tid = threadIdx.x%W;

  __shared__ int start[K];
  __shared__ int end[K];

//  int blocksPerTask = P/tasks;
  int blocksPerTask=0;
  if(blocksPerTask == 0) blocksPerTask=1;
  int myTask = blockIdx.x/blocksPerTask;
  int taskOffset = size*K*myTask;

  if(myTask < tasks) {

    if(threadIdx.x<K) {
      start[threadIdx.x] = pivots[(blockIdx.x*K)+threadIdx.x];
//      if(start[threadIdx.x]%B != 0) // Align to B
//        start[threadIdx.x] = start[threadIdx.x] -start[threadIdx.x]%B;
    }

    if(threadIdx.x<K)
      end[threadIdx.x] = pivots[(P*K)+threadIdx.x];
    if(blockIdx.x % blocksPerTask < blocksPerTask-1 && threadIdx.x < K)
      end[threadIdx.x] = pivots[((blockIdx.x+1)*K)+threadIdx.x];

//      if(end[threadIdx.x]%B != 0) // Align to B
//        end[threadIdx.x] = end[threadIdx.x] -end[threadIdx.x]%B;

    int outputOffset=0;
    for(int i=0; i<K; i++)
      outputOffset+=start[i];
//    printf("block:%d, thread:%d, tasks:%d, myTask:%d, outputOffset:%d\n", blockIdx.x, threadIdx.x, tasks,myTask, outputOffset);
//if(threadIdx.x==0)
//  printf("block:%d, start[0]:%d, end[0]:%d, start[1]:%d, end[1]:%d, start[2]:%d, end[2]:%d, start[3]:%d, end[3]:%d, outputOffset:%d\n", blockIdx.x, start[0], end[0], start[2], end[2], outputOffset);

//if(threadIdx.x==0 && blockIdx.x==0) {
//printf("iters:%d\n", (P+1)*K);
//  for(int i=0; i<(P+1)*K; i++) printf("%d ", pivots[i]); printf("\n");
//}

__syncthreads();
    blockMultimerge<T>(data+taskOffset, output+taskOffset, start, end, size, outputOffset);
  } 
}
*/

