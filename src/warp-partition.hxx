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



template<typename T, fptr_t f>
__device__ void equals(T a, T b) {
  return !(f(a, b) ^ f(b, a));
}

/*
Code to evenly partition a single task into multiple parts, each to be handled by a single warp.

Parameters:
T*              data -- The data to be sorted. The precondition is that every segment of length 'size' (defined as the third parameter) should be sorted.
int*      tempPivots -- The location to store the output. The output for each warp will simply be K different integers, each defining the starting location for that warp's merge.
int             size -- Length of each (sorted) subarray to be merged into one array of size K*size.
int     warpIdInTask -- Index of the warp, relative to the task rather than the block. For all warps with warpIdInTask = 0, the expected output will just be [0, 0, 0 ..., 0] (K 0's).
int edgeCaseTaskSize -- The size of the edge case. In the event that you do not have K filled subarrays, this specifies how large the task size is. For non-edge cases, this variable is just K*size.

Additional Notes:
int L -- In the event of an edge case where you do not have K filled subarrays, you run L-way mergesort. If it is not an edge case, L=K, so simply K-way mergesort.

*/

template<typename T, fptr_t f>
__device__ void warp_partition(T* data, int* tempPivots, int size, int warpsPerTask, int warpIdInTask, int edgeCaseTaskSize) {
  const int WARPS = THREADS/W;
  int tid = threadIdx.x%W;
  int warpInBlock = threadIdx.x/W;
  int targetPivot = ((warpIdInTask)*(edgeCaseTaskSize/warpsPerTask));
  T minVal,maxVal;
  int minIdx, maxIdx;
  int L = (edgeCaseTaskSize + size - 1) / size;

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
          if(f(candidates[iterIdx], minVal) && tempPivots[i] < size-1) {
            minVal = candidates[iterIdx];
            minIdx = i;
          }
          if(f(maxVal, candidates[iterIdx]) && tempPivots[i] > 0) {
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
    if(tid < L) {
      if(tid != partList[warpInBlock]) {
        int left = 0;
        int right = end;
        int mid = end/2;

        while (right - left > 1) {
          if (left == right - 2) {
              mid = left;
          }
          if (f(partVal[warpInBlock], data[size*tid + mid])) {
            right = mid + 1;
            mid = (left+right)/2;
          } else {
            left = mid + 1;
            mid = (left+right)/2;
          }
        }
        tempPivots[tid] = mid;
      }
    }
  }
}

/*
  Finds the pivots for all warps. Invokes warp-partition to properly binary search every pivot.
  Note that this is not optimized code. This method currently involves binary searching every pivot, regardless of whether or not the warp has the task all to itself.

  Parameters:
  T* data     -- The input data to find partitions for. Note that the data is expected to be filled with contiguous segments of length 'size' (defined by third parameter) that are all sorted. 
  T* output   -- The output, where we will store the result of the following merge. We don't actually need this parameter.
  int size    -- The size of each currently sorted subarray
  int tasks   -- The number of tasks we have. Each task is defined as a single merge of K different sorted subarrays of length 'size' into a single subarray of length K*size
  int P       -- The number of blocks we have, equivalent to gridDim.x
  int edgeCaseTaskSize -- The size of the edge case, if there is any. In the event of a non-completely filled task of total length K*size, this will have a value between size < edgeCaseTaskSize < K*size. Otherwise, this will just be = K*size.

*/

// If there is an edge case
template<typename T, fptr_t f>
__global__ void findPartitions(T* data, T*output, int* pivots, int size, int numLists, int tasks, int P, int edgeCaseTaskSize) {
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
  __syncwarp();
  if(myTask < tasks-1) {
    // In this case, we don't have the edge case, so we reuse the same warp_partition function
    warp_partition<T, f>(data+taskOffset, myPivots, size, warpsPerTask, warpIdInTask, K*size);

    
  } else if (myTask == tasks-1) { // It should technically always fall into the else case, but putting if() just in case

    warp_partition<T, f>(data+taskOffset, myPivots, size, warpsPerTask, warpIdInTask, edgeCaseTaskSize);

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
    pivots[warpsPerTask*tasks*K+threadIdx.x] = end;
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
