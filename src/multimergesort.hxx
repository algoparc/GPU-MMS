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
#include "pad.hxx"

// Function that does a single level of multiway mergesort
template<typename T, fptr_t f>
__global__ void multimergeLevel(T* data, T* output, int* pivots, long size, int tasks, int P);
template<typename T, fptr_t f>
__global__ void ml(T* data, T* output, int* pivots, long size, int tasks, int P);
template<typename T, fptr_t f>
__global__ void multimergeLevelEdgeCase(T* data, T* output, int* pivots, long size, int tasks, int P, int N);
template<typename T, fptr_t f>
__global__ void toplevelmerge(T* data, T* output, int numLists, long listSize, long edgeListSize);

template<typename T>
__global__ void copy(T* arr1, T* arr2);

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

#define ERROR_LOGS

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
  cudaError_t err;
  #ifdef ERROR_LOGS
  err = cudaGetLastError();
  if (err != cudaSuccess){
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
  } else {
    printf("No errors immediately after multimergesort function call\n");
  }
  #endif
  int WARPS = P*(THREADS/W);
  int* pivots;
  cudaMalloc((void**) &pivots, (WARPS+1)*K*sizeof(int));
  // Optimization: remove this
  cudaMemset(pivots, 0, (WARPS+1)*K*sizeof(int));
  int tasks;
  T* list[2];
  list[0]=input;
  list[1]=output;
  bool listBit = false;
  int baseBlocks=((N/M)/(THREADS/W));

  // FOR DEBUGGING, YOU CAN DELETE
  T* values = (T*) malloc(N * sizeof(T));

  #ifdef ERROR_LOGS
  printf("CASE 1\n");
  #endif
// Sort the base case into blocks of 1024 elements each
  squareSort<T,f><<<baseBlocks,THREADS>>>(input, N);

// Check that basecase properly sorted if in DEBUG mode
#ifdef DEBUG
  cudaDeviceSynchronize();
  cudaMemcpy(h_data, input, N*sizeof(T), cudaMemcpyDeviceToHost);
  bool correct=true;
  #if PRINT == 1
    printf("[%d", h_data[0]);
    for (int i = 1; i < M; i++)
      printf(", %d", h_data[i]);
    printf("]\n");
  #endif

  for(int i=0; i<N/M; i++) {
    for(int j=1; j<M; j++) {
      if(host_cmp(h_data[i*M+(j)], h_data[i*M+(j-1)])) {
        correct=false;
      }
    }
  }
  if(!correct) printf("base case not sorted!\n");
  else printf("base case was sorted!\n");
#endif

/*
  Perform successive merge rounds.
  We are sorting lists with size listSize, which starts at M=1024,
  because we just finished sorting the base cases of contiguous
  1024 elements.
*/
  int listSize = M;
  #ifdef ERROR_LOGS
  err = cudaGetLastError();
  if (err != cudaSuccess){
    printf("LINE 135 CUDA Error: %s\n", cudaGetErrorString(err));
  } else {
    printf("No errors after squaresort function call\n");
  }
  #endif
  int counter = 0;
  for(listSize=M; listSize < N; listSize *= K) {
    #ifdef ERROR_LOGS
    printf("SORT FOR THIS LEVEL COMPLETED\n");
    testSortedSegments<T, cmp><<<1,1>>>(list[listBit], listSize, N);
    cudaDeviceSynchronize();
    #ifdef ERROR_LOGS
    err = cudaGetLastError();
    if (err != cudaSuccess){
      printf("LINE 148 CUDA Error: %s\n", cudaGetErrorString(err));
    }
    #endif
    printf("-----------------------------------------------------\n");
    #endif
    tasks = N/listSize/K;
    counter++;
    if(tasks > WARPS) { // If each warp has to perform multiple merges
      #ifdef ERROR_LOGS
      printf("CASE 2\n");
      #endif
      for(int i=0; i<tasks/WARPS; i++) {
        findPartitions<T><<<P,THREADS>>>(list[listBit]+(i*WARPS*K*listSize), list[!listBit]+(i*WARPS*K*listSize), pivots, listSize, WARPS*K, WARPS, P); // Why is WARPS*K = numLists? tasks = WARPS because each warp handles a single task
        cudaDeviceSynchronize();

#ifdef DEBUG // Check proper partitioning if debug mode
  testPartitioning<T><<<P,THREADS>>>(list[listBit]+(i*WARPS*K*listSize), pivots, listSize, tasks,WARPS);
#endif

	// Merge based on partitions
        multimergeLevel<T,f><<<P,THREADS>>>(list[listBit]+(i*WARPS*K*listSize), list[!listBit]+(i*WARPS*K*listSize), pivots, listSize, WARPS, P);
      }
      #ifdef ERROR_LOGS
      cudaDeviceSynchronize();
      err = cudaGetLastError();
      if (err != cudaSuccess) {
        printf("LINE 177 ERROR: %s\n", cudaGetErrorString(err));
      } else {
        printf("NO ISSUES, CASE 2 COMPLETED\n");
      }
      #endif

      int edgeCaseTasks = tasks%WARPS;
      int offset = (tasks/WARPS)*WARPS*K*listSize;
      
      int edgeCaseTaskSize = N%(K*listSize);
      edgeCaseTasks += edgeCaseTaskSize > listSize;
      #ifdef ERROR_LOGS
      printf("listSize: %d\n", listSize);
      #endif
      
      if (edgeCaseTaskSize > listSize) {
        #ifdef ERROR_LOGS
        printf("CASE 4\n");
        #endif
        fp<T><<<P,THREADS>>>(list[listBit]+offset, list[!listBit]+offset, pivots, listSize, edgeCaseTasks*K, edgeCaseTasks, P, edgeCaseTaskSize);
        #ifdef ERROR_LOGS
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess) {
          printf("LINE 201 ERROR: %s\n", cudaGetErrorString(err));
        }
        #endif
        ml<T,f><<<P,THREADS>>>(list[listBit]+offset, list[!listBit]+offset, pivots, listSize, edgeCaseTasks, P);
        #ifdef ERROR_LOGS
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess) {
          printf("LINE 209 ERROR: %s\n", cudaGetErrorString(err));
        }
        #endif
      } else {
        
        #ifdef ERROR_LOGS
        printf("CASE 3\n");
        #endif
        findPartitions<T><<<P,THREADS>>>(list[listBit]+offset, list[!listBit]+offset, pivots, listSize, edgeCaseTasks*K, edgeCaseTasks, P);
        #ifdef ERROR_LOGS
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess) {
          printf("%d %d %d LINE 222 ERROR: %s\n", listSize, edgeCaseTasks, P, cudaGetErrorString(err));
        }
        #endif
        multimergeLevel<T,f><<<P,THREADS>>>(list[listBit]+offset, list[!listBit]+offset, pivots, listSize, edgeCaseTasks, P);
        #ifdef ERROR_LOGS
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess) {
          printf("LINE 230 ERROR: %s\n", cudaGetErrorString(err));
        }
        #endif
        // Remember that tasks includes the edgeCaseTasks
        if ((N-tasks*K*listSize)/THREADS)
          copy<T><<<(N-tasks*K*listSize)/THREADS, THREADS>>>(list[listBit]+tasks*K*listSize, list[!listBit]+tasks*K*listSize);
        
      }
      
      
      
    }

    else {
      // Each warp only does one task
      int edgeCaseTaskSize = N%(K*listSize);
      tasks += edgeCaseTaskSize > listSize;
      #ifdef ERROR_LOGS
      printf("listSize: %d\n", listSize);
      #endif
      if (edgeCaseTaskSize > listSize) {
        fp<T><<<P,THREADS>>>(list[listBit], list[!listBit], pivots, listSize, tasks*K, tasks, P, edgeCaseTaskSize);
        #ifdef ERROR_LOGS
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess) {
          printf("LINE 257 ERROR: %s\n", cudaGetErrorString(err));
        }
        #endif
        ml<T,f><<<P,THREADS>>>(list[listBit], list[!listBit], pivots, listSize, tasks, P);
        #ifdef ERROR_LOGS
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess) {
          printf("LINE 265 ERROR: %s\n", cudaGetErrorString(err));
        }
        #endif
      } else {
        #ifdef ERROR_LOGS
        printf("CASE 6\n");
        #endif
        findPartitions<T><<<P,THREADS>>>(list[listBit], list[!listBit], pivots, listSize, tasks*K, tasks, P);
        #ifdef ERROR_LOGS
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess) {
          printf("%d %d %d LINE 277 ERROR: %s\n", listSize, tasks, P, cudaGetErrorString(err));
        }
        #endif
        multimergeLevel<T,f><<<P,THREADS>>>(list[listBit], list[!listBit], pivots, listSize, tasks, P);
        #ifdef ERROR_LOGS
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess) {
          printf("LINE 285 ERROR: %s\n", cudaGetErrorString(err));
        }
        #endif
        if ((N-tasks*K*listSize)/THREADS)
          copy<T><<<(N-tasks*K*listSize)/THREADS, THREADS>>>(list[listBit]+tasks*K*listSize, list[!listBit]+tasks*K*listSize);
      }
    }
    listBit = !listBit; // Switch input/output arrays
    #ifdef ERROR_LOGS
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess){
      printf("LINE 297 CUDA Error: %s\n", cudaGetErrorString(err));
    } else {
      printf("No errors at the end of this iteration of the FOR loop\n");
    }
    #endif
  }

  cudaFree(pivots);

  return list[listBit]; // This returns the array that was last used as output
}


/* Main Kernel to merge groups of K lists */
/* data and output are offset to focus on the current set of tasks to be processed by the warps
P = the number of blocks
*/


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
    }
    __syncwarp();

    if(tid<K) 
      end[tid] = pivots[(totalWarps*K)+tid]; // If tid<K, the right-hand side evaluates to size=listSize.
    if(warpIdx % warpsPerTask < warpsPerTask-1 && tid < K) // Only executes in last kernel launch
      end[tid] = pivots[((warpIdx+1)*K)+tid];

    int outputOffset=0;
    __syncwarp();                            // Required to populate the start[] shared array first
    for(int i=0; i<K; i++)
      outputOffset+=start[i]; 

#ifdef PIPELINE
    multimergePipeline<T,f>(data+taskOffset, output+taskOffset, start, end, size, outputOffset);
#else
    multimerge<T,f>(data+taskOffset, output+taskOffset, start, end, size, outputOffset);
#endif
  } 
}

template<typename T, fptr_t f>
__global__ void ml(T* data, T* output, int* pivots, long size, int tasks, int P) {
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

  if(tid<K) {
    start[tid] = pivots[(warpIdx*K)+tid];
  }
  int outputOffset=0;
  __syncwarp();                            // Required to populate the start[] shared array first
  for(int i=0; i<K; i++)
    outputOffset+=start[i]; 
  __syncwarp();

  if(myTask < tasks-1) {

    if(tid<K) 
      end[tid] = size;
    if(warpIdx % warpsPerTask < warpsPerTask-1 && tid < K)
      end[tid] = pivots[((warpIdx+1)*K)+tid];
    __syncwarp();

    

#ifdef PIPELINE
    multimergePipeline<T,f>(data+taskOffset, output+taskOffset, start, end, size, outputOffset);
#else
    multimerge<T,f>(data+taskOffset, output+taskOffset, start, end, size, outputOffset);
#endif
  } else if (myTask == tasks-1) {
    if (tid < K)
      end[tid] = pivots[(totalWarps*K)+tid];
    __syncwarp();
      
    multimergePipeline<T,f>(data+taskOffset, output+taskOffset, start, end, size, outputOffset);
    
  }
}

template<typename T, fptr_t f>
__global__ void multimergeLevelEdgeCase(T* data, T* output, int* pivots, long size, int tasks, int P, int N) {
  int x = N%(size*K) <= size;
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
  __syncthreads();

  /*
  Check if the incomplete task is just a single, lone list (no merging needed). If
  it is just one list, then eval is true; all the warps just goes to do a full task.
  If not, check the other condition: ensure that this warp's task is NOT the last task.
  */
  int eval = 1;
  if (x > 0){
    eval = 1;
  } else {
    eval = myTask<tasks-1; 
  }

  __syncthreads();
  if (eval){
    if (tid<K){
      start[tid] = 0;
      end[tid] = size;
    }
    __syncwarp();
    multimergePipelineEdgeCaseFull<T,f>(data+taskOffset, output+taskOffset, start, end, size, 0);

  } else {
    
    // L-way mergesort, with L<=K
    int totalTaskSize = N-(tasks-1)*K*size;
    int L = (totalTaskSize+size-1)/size;
    int smallerSize = totalTaskSize-(L-1)*size;
    /*
    if (tid == 0)
      printf("L: %d\n", L);
    */
    if (tid<L){
      start[tid] = 0;
    }
    if (tid<L-1)
      end[tid] = size;
    else if (tid<L)
      end[tid] = smallerSize;
    else if (tid<K)
      end[tid] = 0;
    __syncwarp();
    
    if (tid == 0) {
      printf("%d %d %d %d\n", end[0], end[1], end[2], end[3]);
    }
    
    multimergePipelineEdgeCasePartial<T,f>(data+taskOffset, output+taskOffset, start, end, size, smallerSize, 0, L);
    
  }
}

template<typename T, fptr_t f>
__global__ void toplevelmerge(T* data, T* output, int numLists, long listSize, long edgeListSize){

}

template<typename T>
__global__ void copy(T* arr1, T* arr2){
  arr2[blockIdx.x*THREADS + threadIdx.x] = arr1[blockIdx.x*THREADS + threadIdx.x];
}

// Launch with 1 thread and 1 block
template <typename T, fptr_t f>
__global__ void testSortedSegments(int* d_arr, int segmentSize, int N) {
  #ifdef ERROR_LOGS
  for (int i = 0; i < N; i += segmentSize) {
    for (int j = 1; j < segmentSize; j++) {
      if (i+j >= N) {
        break;
      }
      if (f(d_arr[i+j], d_arr[i+j-1])) {
        printf("UNSORTED AT INDEX %d AND %d\n", i+j-1, i+j);
        return;
      }
    }
  }
  printf("SORTED WITH SEGMENT SIZES : %d\n", segmentSize);
  #endif
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

