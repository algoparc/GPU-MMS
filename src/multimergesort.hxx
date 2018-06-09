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
#include "io-merge-gen.hxx"
#include "warp-partition.hxx"



// Function that does a single level of multiway mergesort
template<typename T, fptr_t f>
__global__ void multimergeLevel(T* data, T* output, int* pivots, long size, int tasks, int P);

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

/* Main CPU function that sorts an input and writes the result to output */
template<typename T, fptr_t f>
T* multimergesort(T* input, T* output, T* h_data, int P, int N) {
  int WARPS = P*(THREADS/W);
  int* pivots;
  cudaMalloc(&pivots, (WARPS+1)*K*sizeof(int));
  cudaMemset(&pivots, 0, (WARPS+1)*K*sizeof(int));
  int tasks;
  T* list[2];
  list[0]=input;
  list[1]=output;
  bool listBit = false;
  int baseBlocks=((N/M)/(THREADS/W));

// Sort the base case into blocks of 1024 elements each
  squareSort<T,f><<<baseBlocks,THREADS>>>(input, N);

// Check that basecase properly sorted if in DEBUG mode
#ifdef DEBUG
  bool correct=true;
  cudaMemcpy(h_data, input, N*sizeof(T), cudaMemcpyDeviceToHost);

  for(int i=0; i<N/M; i++) {
    for(int j=1; j<M; j++) {
      if(f(h_data[i*M+(j)], h_data[i*M+(j-1)])) {
        correct=false;
      }
    }
  }
  if(!correct) printf("base case not sorted!\n");
#endif

// Perform successive merge rounds
  for(int listSize=M; listSize <= (N/K); listSize *= K) {
    tasks = (N/listSize)/K;

    if(tasks > WARPS) { // If each warp has to perform multiple merges
      for(int i=0; i<tasks/WARPS; i++) {
        findPartitions<T><<<P,THREADS>>>(list[listBit]+(i*WARPS*K*listSize), list[!listBit]+(i*WARPS*K*listSize), pivots, listSize, WARPS*K, WARPS, P);

#ifdef DEBUG // Check proper partitioning if debug mode
  testPartitioning<T><<<P,THREADS>>>(list[listBit]+(i*WARPS*K*listSize), pivots, listSize, tasks,WARPS);
        cudaDeviceSynchronize();
#endif

	// Merge based on partitions
        multimergeLevel<T,f><<<P,THREADS>>>(list[listBit]+(i*WARPS*K*listSize), list[!listBit]+(i*WARPS*K*listSize), pivots, listSize, WARPS, P);
      }

// Perform remaining tasks
      if(tasks%WARPS > 0) {
        findPartitions<T><<<P,THREADS>>>(list[listBit]+((tasks/WARPS)*WARPS*K*listSize), list[!listBit]+((tasks/WARPS)*WARPS*K*listSize), pivots, listSize, WARPS*K, WARPS, P);
        cudaDeviceSynchronize();

        multimergeLevel<T,f><<<P,THREADS>>>(list[listBit]+((tasks/WARPS)*WARPS*K*listSize), list[!listBit]+((tasks/WARPS)*WARPS*K*listSize), pivots, listSize, tasks%WARPS, P);
      }
    }

    else { // Each warp only does one task
      findPartitions<T><<<P,THREADS>>>(list[listBit], list[!listBit], pivots, listSize, tasks*K, tasks, P);
#ifdef DEBUG
  testPartitioning<T><<<P,THREADS>>>(list[listBit], pivots, listSize, tasks, WARPS);
#endif
      multimergeLevel<T,f><<<P,THREADS>>>(list[listBit], list[!listBit], pivots, listSize, tasks, P);
    }
    listBit = !listBit; // Switch input/output arrays
  }

  cudaFree(pivots);

  return list[listBit];
}


/* Main Kernel to merge groups of K lists */
template<typename T, fptr_t f>
__global__ void multimergeLevel(T* data, T* output, int* pivots, long size, int tasks, int P) {
  int totalWarps = P*(THREADS/W);
  int warpInBlock = threadIdx.x/W;
  int warpIdx = (blockIdx.x)*(THREADS/W)+warpInBlock;
  int tid = threadIdx.x%W;


  __shared__ int startRaw[K*(THREADS/W)];
  int* start = startRaw+(warpInBlock*K);
  __shared__ int endRaw[K*(THREADS/W)];
  int* end = endRaw+(warpInBlock*K);

  int warpsPerTask = totalWarps/tasks;
  if(warpsPerTask == 0) warpsPerTask=1;
  int myTask = warpIdx/warpsPerTask;
  long taskOffset = size*K*myTask;

  if(myTask < tasks) {

    if(tid<K) {
      start[tid] = pivots[(warpIdx*K)+tid];
//      if(start[tid]%B != 0)
//        start[tid] = start[tid] -start[tid]%B;
    }

    if(tid<K) 
      end[tid] = pivots[(totalWarps*K)+tid];
    if(warpIdx % warpsPerTask < warpsPerTask-1 && tid < K)
      end[tid] = pivots[((warpIdx+1)*K)+tid];

    int outputOffset=0;
    for(int i=0; i<K; i++)
      outputOffset+=start[i];

#ifdef PIPELINE
    multimergePipeline<T,f>(data+taskOffset, output+taskOffset, start, end, size, outputOffset);
#else
    multimerge<T>(data+taskOffset, output+taskOffset, start, end, size, outputOffset);
#endif
  } 
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

