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

// #define PARTITION_ERROR_LOGS
#define BLOCK 1

// Find a set of pivots for a given partition
// The selected pivot will always be between [start, end)
// In other words, start boundary is inclusive, and end boundary is exclusive
template<typename T>
__device__ void warp_partition(T* data, int* tempPivots, int size, int warpsPerTask, int warpIdInTask) {
  const int WARPS = THREADS/W;
  int tid = threadIdx.x%W;
  int warpInBlock = threadIdx.x/W;
  int targetPivot = ((warpIdInTask)*((size*K)/warpsPerTask));
  int numSearchesCompleted = 0;
  T minVal,maxVal;
  int minIdx, maxIdx;

  __shared__ T candidates[K*WARPS];

 volatile  __shared__ int startBoundary[K*WARPS];
 volatile  __shared__ int endBoundary[K*WARPS];
 volatile  __shared__ int completedSearch[K*WARPS];

  // Initialize boundary positions
  if(tid < K) {
    startBoundary[warpInBlock*K + tid] = 0;
    endBoundary[warpInBlock*K + tid] = size;
    tempPivots[tid] = ((warpIdInTask)*(size/warpsPerTask));
    completedSearch[warpInBlock*K + tid] = 0;
  }

// first warp of task begins at start of every list
  if(warpIdInTask == 0 && tid < K) {
    tempPivots[tid] = 0;
  }
  #ifdef PARTITION_ERROR_LOGS
  if (blockIdx.x == BLOCK && tid == 0){
    for (int i=0; i<K; i++) {
      for (int j=0; j<3; j++) {
        printf("%d ", data[size*i + tempPivots[i] + j]);
      }
      printf("\n");
    }
  }
  #endif

// Critical: This cannot just be a __syncwarp() operation, since for initialization, we use a single warp to do it. The other warps must wait for that one warp to initialize the arrays.
__syncwarp();

// find min and max elts of list
  if(warpIdInTask > 0) {
    // Set initial candidate values
    if(tid < K) {
      candidates[warpInBlock*K+tid] = data[size*tid + tempPivots[tid]];
    }
    __syncwarp();

// Sequential section per warp 
    if(tid==0) {
      #ifdef PARTITION_ERROR_LOGS
      if (blockIdx.x == BLOCK) {
          printf("START: %d %d %d %d\n", startBoundary[warpInBlock*K], startBoundary[warpInBlock*K+1], startBoundary[warpInBlock*K+2], startBoundary[warpInBlock*K+3]);
          printf("END: %d %d %d %d\n", endBoundary[warpInBlock*K], endBoundary[warpInBlock*K+1], endBoundary[warpInBlock*K+2], endBoundary[warpInBlock*K+3]);
          printf("PIVOTS: %d %d %d %d\n", tempPivots[0], tempPivots[1], tempPivots[2], tempPivots[3]);
          printf("COMPLETED: %d %d %d %d\n", completedSearch[warpInBlock*K], completedSearch[warpInBlock*K+1], completedSearch[warpInBlock*K+2], completedSearch[warpInBlock*K+3]);
        }
      #endif
      int iterIdx;
      minVal=0;
      maxVal=1;

      while(numSearchesCompleted < K) {
      minVal=MAXVAL;
      maxVal=MINVAL;
      minIdx = -1;
      maxIdx = -1;
        // find min and max - OPTIMIZE use K threads to do min and max reduction
        for(int i=0; i<K; i++) {
          iterIdx = warpInBlock*K + i;
          if(!completedSearch[iterIdx] && cmp(candidates[iterIdx], minVal) && tempPivots[i] < size-1) {
            minVal = candidates[iterIdx];
            minIdx = i;
          }
          if(!completedSearch[iterIdx] && cmp(maxVal, candidates[iterIdx]) && tempPivots[i] > 0) {
            maxVal = candidates[iterIdx];
            maxIdx = i;
          }
        }
        if (minIdx == maxIdx)
          break;

        int minIdxBoundaryShift;
        int maxIdxBoundaryShift;
        int sharedBoundaryShift;

        minIdxBoundaryShift = tempPivots[minIdx] - startBoundary[warpInBlock*K + minIdx];
        maxIdxBoundaryShift = endBoundary[warpInBlock*K + maxIdx] - tempPivots[maxIdx]; // Increase rank of current partition boundary
        #ifdef PARTITION_ERROR_LOGS
        if (blockIdx.x == BLOCK) {
          printf("INDICES: %d %d\n", minIdx, maxIdx);
          printf("SHIFTS: %d %d\n", minIdxBoundaryShift, maxIdxBoundaryShift);
        }
        #endif

        if (minIdxBoundaryShift == 0) {
          completedSearch[warpInBlock*K + minIdx] = 1;
          numSearchesCompleted++;
          continue;
        }
        if (maxIdxBoundaryShift == 0) {
          completedSearch[warpInBlock*K + maxIdx] = 1;
          numSearchesCompleted++;
          continue;
        }
        if (minIdxBoundaryShift > maxIdxBoundaryShift) {
          sharedBoundaryShift = maxIdxBoundaryShift;
        } else {
          sharedBoundaryShift = minIdxBoundaryShift;
        }
        
        startBoundary[warpInBlock*K + minIdx] += sharedBoundaryShift;
        endBoundary[warpInBlock*K + maxIdx] -= sharedBoundaryShift;
        // END OF CHANGES
        int minIdxPivotShift = 0; 
        tempPivots[minIdx]=(endBoundary[warpInBlock*K+minIdx]+startBoundary[warpInBlock*K+minIdx])/2;
        tempPivots[maxIdx]=(endBoundary[warpInBlock*K+maxIdx]+startBoundary[warpInBlock*K+maxIdx])/2;
        
        candidates[warpInBlock*K + minIdx] = data[size*minIdx + tempPivots[minIdx]];
        if(startBoundary[warpInBlock*K + minIdx] >= endBoundary[warpInBlock*K + minIdx]-1 && tempPivots[minIdx] < size) {
          numSearchesCompleted++;
          completedSearch[warpInBlock*K + minIdx] = 1;
        } // Increase rank of current partition boundary
        candidates[warpInBlock*K + maxIdx] = data[size*maxIdx + tempPivots[maxIdx]];
        if(startBoundary[warpInBlock*K + maxIdx] >= endBoundary[warpInBlock*K + maxIdx]-1 && tempPivots[maxIdx] > 0) {
          numSearchesCompleted++;
          completedSearch[warpInBlock*K + maxIdx] = 1;
        }
        #ifdef PARTITION_ERROR_LOGS
        if (blockIdx.x == BLOCK) {
          printf("START: %d %d %d %d\n", startBoundary[warpInBlock*K], startBoundary[warpInBlock*K+1], startBoundary[warpInBlock*K+2], startBoundary[warpInBlock*K+3]);
          printf("END: %d %d %d %d\n", endBoundary[warpInBlock*K], endBoundary[warpInBlock*K+1], endBoundary[warpInBlock*K+2], endBoundary[warpInBlock*K+3]);
          printf("PIVOTS: %d %d %d %d\n", tempPivots[0], tempPivots[1], tempPivots[2], tempPivots[3]);
          printf("COMPLETED: %d %d %d %d\n", completedSearch[warpInBlock*K], completedSearch[warpInBlock*K+1], completedSearch[warpInBlock*K+2], completedSearch[warpInBlock*K+3]);
        }
        #endif

      }
      
    }
    
  }
  __syncthreads();
  if (warpIdInTask > 0 && tid < K) {
    if (tempPivots[tid] >= size) {
      tempPivots[tid] = size-1;
    } else {
      tempPivots[tid]++; // We need to add 1, because we have the property that pivot[i] <= pivot[j+1] for all pivots
    }
    #ifdef PARTITION_ERROR_LOGS
    if (blockIdx.x == BLOCK && tid == 0){
      for (int i=0; i<K; i++) {
        for (int j=-2; j<3; j++) {
          printf("%d ", data[size*i + tempPivots[i] + j]);
        }
        printf("\n");
      }
    }
    #endif
  }
  __syncwarp();
  // Make slight adjustments as needed
  /*
  if (tid == 0){
    int changed = 0;
    do {
      changed = 0;
      for (int i=0; i<K; i++) {
        for (int j=1; j<K; j++) {
          if (data[size*i + tempPivots[i] - 1] > data[size*j + tempPivots[j]]) {
            tempPivots[j]++;
            changed = 1;
          }
        }
      }
    } while (changed);

  }
  */
}

template<typename T>
__device__ void wp(T* data, int* tempPivots, int size, int warpsPerTask, int warpIdInTask, int edgeCaseTaskSize) {
}

// Find pivots K pivots for each warp within a 'task' (a group of K lists)
// Pivots define the start of the partition that each warp will work on merging
// P = Number of blocks in grid
template<typename T>
__global__ void findPartitions(T* data, T*output, int* pivots, int size, int numLists, int tasks, int P) {
  __shared__ int myPivotsRaw[K*(THREADS/W)];
  int warpInBlock = threadIdx.x/W;
  int* myPivots = myPivotsRaw+(warpInBlock*K);
  int warpIdx = (blockIdx.x)*(THREADS/W) + warpInBlock;
  int tid = threadIdx.x%W;
  int warpsPerTask;
  int myTask;
  int taskOffset;
  int warpIdInTask;
  int totalWarps = P*(THREADS/W);
  
  warpsPerTask = totalWarps/tasks; // floor
      //if (blockIdx.x == 0) {
      //  printf("tasks: %d\n", tasks);
      //}
  if(warpsPerTask <= 1) {
    if(tid < K) {
      pivots[warpIdx*K+tid] = 0;
    }
    if(blockIdx.x==0 && threadIdx.x < K) {
      pivots[totalWarps*K+threadIdx.x] = size;
    }
  }
  else {
    myTask = warpIdx / warpsPerTask; // If we have extra warps, just have them do no work...
    if(myTask < tasks) {
      taskOffset = myTask*size*K;
      warpIdInTask = warpIdx - myTask*warpsPerTask;

      warp_partition<T>(data+taskOffset, myPivots, size, warpsPerTask, warpIdInTask);
      __syncwarp();
      if(tid < K) {
        pivots[warpIdx*K+tid] = myPivots[tid];
      }
      if(blockIdx.x ==0 && threadIdx.x<K) {  // Fill last K spots with max val
        pivots[totalWarps*K+threadIdx.x] = size;
      }
    }
  }
}

// If there is an edge case
template<typename T>
__global__ void fp(T* data, T*output, int* pivots, int size, int numLists, int tasks, int P, int edgeCaseTaskSize) {
  __shared__ int myPivotsRaw[K*(THREADS/W)];
  int warpInBlock = threadIdx.x/W;
  int* myPivots = myPivotsRaw+(warpInBlock*K);
  int warpIdx = (blockIdx.x)*(THREADS/W) + warpInBlock;
  int tid = threadIdx.x%W;
  int warpsPerTask;
  int myTask;
  int taskOffset;
  int warpIdInTask;
  int totalWarps = P*(THREADS/W);

  /*
    In the general case, warpsPerTask = totalWarps / WARPS = (P * THREADS / W) / (P * THREADS/W) = 1
    Edge case: warpsPerTask = totalWarps / ... = (P * THREADS / W) / ... >= 1
  */
  warpsPerTask = totalWarps/tasks;
  myTask = warpIdx / warpsPerTask; // If we have extra warps, just have them do no work...
  taskOffset = myTask*size*K;
  warpIdInTask = warpIdx - myTask*warpsPerTask;
  if(myTask < tasks-1) {
    // In this case, we don't have the edge case, so we reuse the same warp_partition function
    warp_partition<T>(data+taskOffset, myPivots, size, warpsPerTask, warpIdInTask);

    
  } else if (myTask == tasks-1) { // It should technically always fall into the else case, but putting if() just in case
    // TODO: Implement wp
    wp<T>(data+taskOffset, myPivots, size, warpsPerTask, warpIdInTask, edgeCaseTaskSize);


    //
  }
  if(tid < K) {
    pivots[warpIdx*K+tid] = myPivots[tid];
  }
  if(blockIdx.x==0 && threadIdx.x<K) {  // Fill last K spots with end values
    int difference = edgeCaseTaskSize - threadIdx.x * size;
    int end = (difference < size)*difference + (size <= difference)*size; // taking MIN of difference and size using predicates
    pivots[totalWarps*K+threadIdx.x] = end;
  }
}

/* FOR DEBUGGING - MAKES SURE PIVOTS MAKE A VALID PARTITION */
template<typename T>
void __global__ testPartitioning(T* data, int* pivots, int size, int tasks, int P) {
  int warpsPerTask;
  int totalWarps = P*(THREADS/W);
  warpsPerTask = totalWarps/tasks; // floor

  if(threadIdx.x==0 && blockIdx.x==0) {
    int error=false;
    int pivotVal;
  for(int i=1; i<warpsPerTask; i++) {
  for(int j=0; j<K; j++) {
    if(pivots[i*K+j] < size) {
      pivotVal=data[size*j + pivots[i*K+j]];
      for(int k=0; k<K; k++) {
        if(pivots[i*K + k] > 0 && pivotVal < data[size*k + pivots[i*K + k] -1]) {
          error=true;
        }
      }
    }
  }
  for(int i=0; i<K*warpsPerTask;i++)
    printf("%d ", pivots[i]);
  }
    if(error)
      printf("Partitioning failed\n");
    else
      printf("Partitions correct!\n");
  }
}
