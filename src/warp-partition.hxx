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
__device__ void warp_partition(T* data, int* tempPivots, int size, int blocksPerTask, int blockIdInTask, int edgeCaseTaskSize) {
  int targetPivot = ((blockIdInTask)*(edgeCaseTaskSize/blocksPerTask));
  T minVal,maxVal;
  int minIdx, maxIdx;
  int L = (edgeCaseTaskSize + size - 1) / size;
  if (L == 1) {
    if (threadIdx.x == 0)
      tempPivots[0] = ((blockIdInTask)*(edgeCaseTaskSize/blocksPerTask));
    else if (threadIdx.x < K)
      tempPivots[threadIdx.x] = 0;
    return;
  }

  __shared__ T candidates[K];
  __shared__ int pivotIdxSum;

 volatile  __shared__ int startBoundary[K];
 volatile  __shared__ int endBoundary[K];

  int end = 0;
  __syncthreads();
  // Initialize boundary positions
  if(threadIdx.x < K) {

    startBoundary[threadIdx.x] = 0;
    int difference = edgeCaseTaskSize - threadIdx.x*size;
    if (difference >= 0) {
      end = (difference < size)*difference + (size <= difference)*size; // Minimum of difference and size, using predicate logic
    }
    endBoundary[threadIdx.x] = end;
    tempPivots[threadIdx.x] = end/2;
  }
  __syncwarp();

  if(threadIdx.x==0) {
    int sum = 0;
    for (int i = 0; i < K; i++) {
      sum += tempPivots[i];
    }
    pivotIdxSum = sum;
  }
  __syncwarp();


// first warp of task begins at start of every list
  if(blockIdInTask == 0 && threadIdx.x < K) {
    tempPivots[threadIdx.x] = 0;
  }

  __syncwarp();

// find min and max elts of list
  if(blockIdInTask > 0) {
    // Set initial candidate values
    if(threadIdx.x < L) {
      candidates[threadIdx.x] = data[size*threadIdx.x + tempPivots[threadIdx.x]];
    }
    

    __shared__ T partitionVal;
    __shared__ int partitionIdx;
    __syncwarp();
// Sequential section per warp 
    if(threadIdx.x==0) {
      
      
      partitionVal=MAXVAL;
      while(partitionVal == MAXVAL) {
      minVal=MAXVAL;
      maxVal=MINVAL;

          
        for(int i=0; i<L; i++) {
          if(f(candidates[i], minVal) && tempPivots[i] < size-1) {
            minVal = candidates[i];
            minIdx = i;
          }
          if(f(maxVal, candidates[i]) && tempPivots[i] > 0) {
            maxVal = candidates[i];
            maxIdx = i;
          }
        }

// Move min and/or max candidates based on target order statistic
        

          // Check if the target pivot is greater than the current sum of pivot indices. If it is, move the minimum of the pivot indices
        if(targetPivot >= pivotIdxSum) {
          pivotIdxSum += (endBoundary[minIdx] - tempPivots[minIdx])/2; // Increase rank of current partition boundary
          startBoundary[minIdx] = tempPivots[minIdx];
          tempPivots[minIdx]=(endBoundary[minIdx]+startBoundary[minIdx])/2;
          if(tempPivots[minIdx] == startBoundary[minIdx]) { // Edge case, when start=pivot=end-1
            tempPivots[minIdx]++; 
            pivotIdxSum++;
          }
          candidates[minIdx] = data[size*minIdx + tempPivots[minIdx]];
          if(startBoundary[minIdx] >= endBoundary[minIdx] && tempPivots[minIdx] < size-1)  // If start >= end, we are done finding the partition value
            partitionVal = candidates[minIdx];
          partitionIdx = minIdx;
        } 
        else { // Check if the target 
          pivotIdxSum -= (tempPivots[maxIdx] - startBoundary[ maxIdx])/2; // Increase rank of current partition boundary
          endBoundary[maxIdx] = tempPivots[maxIdx];
          tempPivots[maxIdx]=(endBoundary[maxIdx]+startBoundary[maxIdx])/2;
          candidates[maxIdx] = data[size*maxIdx + tempPivots[maxIdx]];
          if(startBoundary[maxIdx] >= endBoundary[maxIdx] && tempPivots[maxIdx] > 0)  // If start >= end, we are done finding the partition value
            partitionVal = candidates[maxIdx];
          partitionIdx = maxIdx;
        }
        
      }
    }
      // Binary search each other list to find predecessor of partitioning value
    __syncwarp();

    if(threadIdx.x < L) {
      if(threadIdx.x != partitionIdx) {
        int left = 0;
        int right = end;
        int mid = end/2;

        while (right - left > 1) {
          if (left == right - 2) {
              mid = left;
          }
          if (f(partitionVal, data[size*threadIdx.x + mid])) {
            right = mid + 1;
            mid = (left+right)/2;
          } else {
            left = mid + 1;
            mid = (left+right)/2;
          }
        }
        tempPivots[threadIdx.x] = mid;
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
__global__ void findPartitions(T* data, int* pivots, int size, int tasks, int edgeCaseTaskSize) {
  __shared__ int myPivots[K];
  int myTask;
  int taskOffset;
  int blocksPerTask;
  int blockIdInTask;

  blocksPerTask = gridDim.x/tasks;
  myTask = blockIdx.x / blocksPerTask; // If we have extra warps, just have them do no work...
  taskOffset = myTask*size*K;
  blockIdInTask = blockIdx.x - myTask*blocksPerTask;
  __syncthreads();
  if(myTask < tasks-1) {
    // In this case, we don't have the edge case, so we reuse the same warp_partition function
    warp_partition<T, f>(data+taskOffset, myPivots, size, blocksPerTask, blockIdInTask, K*size);

    
  } else if (myTask == tasks-1) {
    warp_partition<T, f>(data+taskOffset, myPivots, size, blocksPerTask, blockIdInTask, edgeCaseTaskSize);

  }
  __syncthreads();
  if(threadIdx.x < K) {
    pivots[blockIdx.x*K+threadIdx.x] = myPivots[threadIdx.x];
  }
  // Not sure why this wouldn't be done by the last block instead of first block... Oh wait it doesn't matter. But for readability, make it the last block.
  if(myTask==tasks-1 && threadIdx.x<K) {  // Fill last K spots with end values
    int difference = edgeCaseTaskSize - threadIdx.x * size;
    difference = (difference > 0) * difference;
    int end = (difference < size)*difference + (size <= difference)*size; // taking MIN of difference and size using predicates
    pivots[blocksPerTask*tasks*K+threadIdx.x] = end;
  }
  
}

void __global__ printPartitions(int* pivots, int blocks) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("-----------------------------------------------------------------------\n");
    for (int i=0; i<K*blocks+K; i++) {
      printf("%d ", pivots[i]);
    }
    printf("\n");
  }
  
}

/* FOR DEBUGGING - MAKES SURE PIVOTS MAKE A VALID PARTITION */
template<typename T>
void __global__ testPartitioning(T* data, int* pivots, int size, int tasks, int P) {
  int blocksPerTask = gridDim.x/tasks; // floor

  if(threadIdx.x==0 && blockIdx.x==0) {
    int pivotVal;
    int errorLocation=-1;
    for(int i=1; i<blocksPerTask*tasks; i++) {
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
