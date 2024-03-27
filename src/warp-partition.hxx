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
__device__ int equals(T a, T b) {
  return !(f(a, b) ^ f(b, a));
}

__global__ void defaultPartitions(int* pivots, int P, int size, int edgeCaseTaskSize) {
  int globalThreadIdx = blockDim.x * blockIdx.x + threadIdx.x;
  if (globalThreadIdx < K*P) {
    pivots[globalThreadIdx] = 0;
    if (globalThreadIdx < K) {
      int difference = edgeCaseTaskSize - threadIdx.x * size;
      difference = (difference > 0) * difference;
      int end = (difference < size)*difference + (size <= difference)*size; // taking MIN of difference and size using predicates
      pivots[P*K+threadIdx.x] = end;
    }
  }
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
__device__ void warp_partition(T* data, int* tempPivots, long size, long warpsPerTask, long warpIdInTask, long edgeCaseTaskSize) {
  int tid = threadIdx.x%W;
  if (warpIdInTask == 0) {
    if (tid < K) {
      tempPivots[tid] = 0;
    }
  } else if (tid == 0) {
    int left[K];
    int right[K];
    int mid[K];
    int completed[K];
    int totalCompleted=0;
    int L = (edgeCaseTaskSize+size-1)/size;
    int sum=0;
    long target = warpIdInTask*edgeCaseTaskSize/warpsPerTask;
    edgeCaseTaskSize--;
    for (int i=0; i<L; i++) {
      left[i] = -1;
      if (edgeCaseTaskSize > 0) {
        right[i] = (edgeCaseTaskSize > size) ? size : edgeCaseTaskSize;
      } else {
        right[i] = 0;
      }
      mid[i] = right[i]/2;
      sum += mid[i];
      edgeCaseTaskSize -= size;
      completed[i] = 0;
    }

    int idx=0;

    while (totalCompleted < L-1) {
      int firstIndex;
      for (int i=L-1; i>=0; i--) {
        if (!completed[i]) {
          firstIndex = i;
        }
      }
      T val = data[size*firstIndex + mid[firstIndex]];
      int moveMax = sum >= target;
      idx = firstIndex;
      for (int i=1; i<L; i++) {
        if (!completed[i]) {
          if (moveMax && (equals<T,f>(val, data[size*i + mid[i]]) || f(val, data[size*i + mid[i]]))) {
            idx = i;
            val = data[size*i + mid[i]];
          } else if (!moveMax && f(data[size*i + mid[i]], val)) {
            idx = i;
            val = data[size*i + mid[i]];
          }
        }
      }
      
      if (moveMax) {
        right[idx] = mid[idx];
      } else {
        left[idx] = mid[idx];
      }
      sum -= mid[idx];
      mid[idx] = (left[idx]+right[idx])/2;
      sum += mid[idx];
      
      if (left[idx] + 1 == right[idx]) {
        totalCompleted++;
        completed[idx] = 1;
        sum++; // Add 1 to sum, because mid currently equals left, and we return right[];
      } else if (left[idx] + 1 > right[idx]) {
        printf("LOOP INVARIANT VIOLATED! LEFT[IDX] + 1 > RIGHT[IDX]\n");
        return;
      }
    }

    for (int i=0; i<K; i++) {
      if (i<L) {
	if (!completed[i]) {
    mid[i] += target-sum;
	  tempPivots[i] = mid[i];
	} else {
          tempPivots[i] = right[i];
	}
      } else {
        tempPivots[i] = 0;
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

template<typename T, fptr_t f>
__global__ void findPartitions(T* data, T*output, int* pivots, long size, int numLists, int tasks, int P, int edgeCaseTaskSize) {
  __shared__ int myPivotsRaw[K*(THREADS/W)];
  int warpInBlock = threadIdx.x/W;
  int* myPivots = myPivotsRaw+(warpInBlock*K);
  int warpIdx = (blockIdx.x)*(THREADS/W) + warpInBlock;
  int tid = threadIdx.x%W;
  long warpsPerTask;
  int myTask;
  int taskOffset;
  long warpIdInTask;
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
      if (i % K == 0) {
        printf("| ");
      }
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
    for (int z=0; z<tasks; z++){
      for(int i=0; i<warpsPerTask; i++) {
        int offset = z*warpsPerTask+i;
        for(int j=0; j<K; j++) {
          if (pivots[offset+j] > 0) {
            pivotVal=data[size*offset + pivots[offset+j]];
            for(int k=0; k<K; k++) {
              if(pivots[offset + k] > 0 && pivotVal < data[size*offset + pivots[offset + k] -1]) {
                printf("i,j,k equal %d %d %d\n", i,j,k);
                errorLocation=size*k + pivots[offset + k] - 1;
                int p1 = size*offset+pivots[offset+j];
                int p2 = size*offset+pivots[offset+k]-1;
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
