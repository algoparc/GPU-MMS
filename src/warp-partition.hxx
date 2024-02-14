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


// Find a set of pivots for a given partition
template<typename T>
__device__ void warp_partition(T* data, int* tempPivots, int size, int warpsPerTask, int warpIdInTask) {
  const int WARPS = THREADS/W;
  int tid = threadIdx.x%W;
  int warpInBlock = threadIdx.x/W;
  int targetPivot = ((warpIdInTask)*((size*K)/warpsPerTask));
  T minVal,maxVal;
  int minIdx, maxIdx;

  __shared__ T sharedCandidates[K*WARPS];
  __shared__ int partitionVal[WARPS];

  if(threadIdx.x < WARPS) {
    partitionVal[threadIdx.x] = (size*K)/2;
  }

 volatile  __shared__ int start[K*WARPS];
 volatile  __shared__ int end[K*WARPS];

  // Initialize boundary positions
  if(threadIdx.x < K*WARPS) {
    start[threadIdx.x] = 0;
    end[threadIdx.x] = size-1;
    tempPivots[tid] = size/2;
  }
  __syncthreads();

 volatile int* startBoundary = start            + K*warpInBlock;
 volatile int* endBoundary   = end              + K*warpInBlock;
 volatile T*   candidates    = sharedCandidates + K*warpInBlock;

// first warp of task begins at start of every list
  if(warpIdInTask == 0 && tid < K) {
    tempPivots[tid] = 0;
  }

__syncwarp();

/*----------------------------------

See Figure 1 in 'figures' directory

-----------------------------------*/

// find min and max elts of list
  if(warpIdInTask > 0) {
    // Set initial candidate values
    if(tid < K) {
      candidates[tid] = data[size*tid + tempPivots[tid]];
    }
    __syncwarp();

    __shared__ T partVal[THREADS/W];
    __shared__ int partList[THREADS/W];
// Sequential section per warp 
    if(tid==0) {
      partVal[warpInBlock]=MAXVAL;
      int tempPartitionVal;
      int iterIdx;
      minVal=0;
      maxVal=1;

      while(partVal[warpInBlock] == MAXVAL) {
      minVal=MAXVAL;
      maxVal=MINVAL;
        // find min and max - OPTIMIZE use K threads to do min and max reduction
        for(int i=0; i<K; i++) {
          iterIdx = i;
          if(cmp(candidates[iterIdx], minVal) && tempPivots[i] < size-1) {
            minVal = candidates[iterIdx];
            minIdx = i;
          }
          if(cmp(maxVal, candidates[iterIdx]) && tempPivots[i] > 0) {
            maxVal = candidates[iterIdx];
            maxIdx = i;
          }
        }
// Move min and/or max candidates based on target order statistic
        tempPartitionVal = partitionVal[warpInBlock];

          // If we need to move min candidate
        if(targetPivot >= tempPartitionVal) {
          partitionVal[warpInBlock] += (endBoundary[minIdx] - tempPivots[minIdx])/2; // Increase rank of current partition boundary
          startBoundary[warpInBlock*K + minIdx] = tempPivots[minIdx];
          tempPivots[minIdx]=(endBoundary[minIdx]+startBoundary[minIdx])/2;
          if(tempPivots[minIdx] == startBoundary[minIdx]) { // Edge case
            tempPivots[minIdx]++; 
            partitionVal[warpInBlock]++;
          }
          candidates[minIdx] = data[size*minIdx + tempPivots[minIdx]];
          if(startBoundary[minIdx] >= endBoundary[minIdx] && tempPivots[minIdx] < size-1) 
//          {
            partVal[warpInBlock] = candidates[minIdx];
            partList[warpInBlock] = minIdx;
//          }
        } 
        else { // If we need to move max candidate
          partitionVal[warpInBlock] -= (tempPivots[maxIdx] - startBoundary[maxIdx])/2; // Increase rank of current partition boundary
          endBoundary[maxIdx] = tempPivots[maxIdx];
          tempPivots[maxIdx]=(endBoundary[maxIdx]+startBoundary[maxIdx])/2;
          candidates[maxIdx] = data[size*maxIdx + tempPivots[maxIdx]];
          if(startBoundary[maxIdx] >= endBoundary[maxIdx] && tempPivots[maxIdx] > 0) 
 //         {
            partVal[warpInBlock] = candidates[warpInBlock*K + maxIdx];
            partList[warpInBlock] = maxIdx;
          }
        }
      }
      // Binary search each other list to find predecessor of partitioning value
    __syncwarp();

    int step;
    if(tid < K) {
      if(tid != partList[warpInBlock]) {
        tempPivots[tid] = size/2;
        step = size/4;

        while(step >= 1) {
          if(!cmp((data[size*tid + tempPivots[tid]]), partVal[warpInBlock])) 
            tempPivots[tid] -= step;
          else 
            tempPivots[tid] += step;
          step /=2;
        }
        if(tempPivots[tid] > 0 && cmp(partVal[warpInBlock], (data[size*tid + tempPivots[tid]-1])))
          tempPivots[tid]--;
        if(cmp((data[size*tid + tempPivots[tid]]), partVal[warpInBlock]))
          tempPivots[tid]++;
      }
    }
  }
}

template<typename T>
__device__ void wp(T* data, int* tempPivots, int size, int warpsPerTask, int warpIdInTask, int edgeCaseTaskSize) {
  const int WARPS = THREADS/W;
  int tid = threadIdx.x%W;
  int warpInBlock = threadIdx.x/W;
  int targetPivot = ((warpIdInTask)*(edgeCaseTaskSize/warpsPerTask));
  T minVal,maxVal;
  int minIdx, maxIdx;

  __shared__ T candidates[K*WARPS];
  __shared__ int partitionVal[WARPS];

 volatile  __shared__ int startBoundary[K*WARPS];
 volatile  __shared__ int endBoundary[K*WARPS];

  int end = 0;
  __syncwarp();
  // Initialize boundary positions
  if(tid < K) {

    startBoundary[warpInBlock*K + tid] = 0;
    int difference = edgeCaseTaskSize - tid*size;
    if (difference >= 0) {
      end = (difference < size)*difference + (size <= difference)*size; // Minimum of difference and size, using predicate logic
    }
    endBoundary[warpInBlock*K + tid] = end;
    tempPivots[tid] = end/2;
  }
  __syncwarp();

  if(tid==0) {
    int sum = 0;
    for (int i = 0; i < K; i++) {
      sum += tempPivots[i];
    }
    partitionVal[warpInBlock] = sum;
  }
  __syncwarp();

  // Edge case 

// first warp of task begins at start of every list
  if(warpIdInTask == 0 && tid < K) {
    tempPivots[tid] = 0;
  }

  __syncwarp();

// find min and max elts of list
  if(warpIdInTask > 0) {
    // Set initial candidate values
    if(tid < K && tempPivots[tid] > 0) {
      candidates[warpInBlock*K+tid] = data[size*tid + tempPivots[tid]];
    }
    

    __shared__ T partVal[THREADS/W];
    __shared__ int partList[THREADS/W];
    __syncwarp();
    
    int L = 0;   // L represents L-way mergesort instead of K-way
    for (int i = 1; i <= K; i++) {
      if (tempPivots[i-1] > 0) {
        L = i;
      }
    }
    __syncwarp();
// Sequential section per warp 
    if(tid==0) {
      
      
      partVal[warpInBlock]=MAXVAL;
      int tempPartitionVal;
      int iterIdx;
      minVal=0;
      maxVal=1;
      while(partVal[warpInBlock] == MAXVAL) {
      minVal=MAXVAL;
      maxVal=MINVAL;

          
        for(int i=0; i<L; i++) {
          iterIdx = warpInBlock*K + i;
          if(cmp(candidates[iterIdx], minVal) && tempPivots[i] < size-1) {
            minVal = candidates[iterIdx];
            minIdx = i;
          }
          if(cmp(maxVal, candidates[iterIdx]) && tempPivots[i] > 0) {
            maxVal = candidates[iterIdx];
            maxIdx = i;
          }
        }
// Move min and/or max candidates based on target order statistic
        tempPartitionVal = partitionVal[warpInBlock];
        

          // If we need to move min candidate
        if(targetPivot >= tempPartitionVal) {
          partitionVal[warpInBlock] += (endBoundary[warpInBlock*K + minIdx] - tempPivots[minIdx])/2; // Increase rank of current partition boundary
          startBoundary[warpInBlock*K + minIdx] = tempPivots[minIdx];
          tempPivots[minIdx]=(endBoundary[warpInBlock*K+minIdx]+startBoundary[warpInBlock*K+minIdx])/2;
          if(tempPivots[minIdx] == startBoundary[warpInBlock*K + minIdx]) { // Edge case
            tempPivots[minIdx]++; 
            partitionVal[warpInBlock]++;
          }
          candidates[warpInBlock*K + minIdx] = data[size*minIdx + tempPivots[minIdx]];
          if(startBoundary[warpInBlock*K + minIdx] >= endBoundary[warpInBlock*K + minIdx] && tempPivots[minIdx] < size-1) 
            partVal[warpInBlock] = candidates[warpInBlock*K + minIdx];
          partList[warpInBlock] = minIdx;
        } 
        else { // If we need to move max candidate
          partitionVal[warpInBlock] -= (tempPivots[maxIdx] - startBoundary[warpInBlock*K + maxIdx])/2; // Increase rank of current partition boundary
          endBoundary[warpInBlock*K + maxIdx] = tempPivots[maxIdx];
          tempPivots[maxIdx]=(endBoundary[warpInBlock*K+maxIdx]+startBoundary[warpInBlock*K+maxIdx])/2;
          candidates[warpInBlock*K + maxIdx] = data[size*maxIdx + tempPivots[maxIdx]];
          if(startBoundary[warpInBlock*K + maxIdx] >= endBoundary[warpInBlock*K + maxIdx] && tempPivots[maxIdx] > 0) 
            partVal[warpInBlock] = candidates[warpInBlock*K + maxIdx];
          partList[warpInBlock] = maxIdx;
        }
        
      }
    }
      // Binary search each other list to find predecessor of partitioning value
    __syncwarp();

    /*
    if (tid < K && tempPivots[tid] >= end) {
      tempPivots[tid] = end-1;
    }
    */
    int step;
    if(tid < L) {
      if(tid != partList[warpInBlock]) {
        tempPivots[tid] = end/2;            // end/2 as opposed to size/2
        step = end/4;

        while(step >= 1) {
          if(!cmp((data[size*tid + tempPivots[tid]]), partVal[warpInBlock])) // if pivot value is greater than or equal to predecessor
            tempPivots[tid] -= step;
          else 
            tempPivots[tid] += step;
          step /=2;
        }
        while(tempPivots[tid] > 0 && cmp(partVal[warpInBlock], (data[size*tid + tempPivots[tid]-1])))
          tempPivots[tid]--;
        while(cmp((data[size*tid + tempPivots[tid]]), partVal[warpInBlock]))
          tempPivots[tid]++;
      }
    }
  }
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
  if (threadIdx.x == 0 && blockIdx.x == 0 && size == 1048576)
    printf("VALS: %d %d %d\n", warpsPerTask, tasks, tasks*warpsPerTask);
  __syncwarp();
  if(myTask < tasks-1) {
    // In this case, we don't have the edge case, so we reuse the same warp_partition function
    warp_partition<T>(data+taskOffset, myPivots, size, warpsPerTask, warpIdInTask);

    
  } else if (myTask == tasks-1) { // It should technically always fall into the else case, but putting if() just in case

    wp<T>(data+taskOffset, myPivots, size, warpsPerTask, warpIdInTask, edgeCaseTaskSize);

  }
  __syncwarp();
  if(tid < K) {
    pivots[warpIdx*K+tid] = myPivots[tid];
  }
  __syncthreads();
  if(blockIdx.x==0 && threadIdx.x<K) {  // Fill last K spots with end values
    int difference = edgeCaseTaskSize - threadIdx.x * size;
    difference = (difference > 0) * difference;
    int end = (difference < size)*difference + (size <= difference)*size; // taking MIN of difference and size using predicates
    pivots[totalWarps*K+threadIdx.x] = end;
  }
  
}

void __global__ printPartitions(int* pivots, int size, int tasks, int P) {
  int totalWarps = P*(THREADS/W);
  int warpsPerTask = totalWarps/tasks;
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    for (int i=0; i<K*warpsPerTask*tasks+K; i++) {
      printf("%d ", pivots[i]);
    }
  }
  printf("\n");
  
}

/* FOR DEBUGGING - MAKES SURE PIVOTS MAKE A VALID PARTITION */
template<typename T>
void __global__ testPartitioning(T* data, int* pivots, int size, int tasks, int P) {
  int totalWarps = P*(THREADS/W);
  int warpsPerTask = totalWarps/tasks; // floor

  if(threadIdx.x==0 && blockIdx.x==0) {
    int pivotVal;
    int errorLocation=-1;
    for(int i=1; i<warpsPerTask*tasks; i++) {
      for(int j=0; j<K; j++) {
        if(pivots[i*K+j] < size) {
          if (pivots[i*K+j] > 0) {
            pivotVal=data[size*j + pivots[i*K+j]];
            for(int k=0; k<K; k++) {
              if(pivots[i*K + k] > 0 && pivotVal < data[size*k + pivots[i*K + k] -1]) {
                printf("i,j,k equal %d %d %d\n", i,j,k);
                errorLocation=size*k + pivots[i*K + k] - 1;
                int p1 = size*j+pivots[i*K+j];
                int p2 = size*k+pivots[i*K+k]-1;
                printf("Partitioning failed, error location : %d\n", errorLocation);
                printf("Neighborhood 1: %d %d %d\n", data[p1-1], data[p1], data[p1+1]);
                printf("Neighborhood 2: %d %d %d\n", data[p2-1], data[p2], data[p2+1]);
                return;
              }
            }
          }
        }
      }
    }
    printf("Partitions correct!\n");
  }
}
