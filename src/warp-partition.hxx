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

#define ERROR_BLOCK 1008

// order = 0 indicates smallest value, order = L-1 indicates largest
template<typename T>
struct Pair {
  int order;
  T val;
};

template<typename T, fptr_t f>
__device__ int equals(T a, T b) {
  return !(f(a, b) ^ f(b, a));
}

// Old partitioning scheme
template<typename T, fptr_t f>
__device__ void old_partition(T* data, int* tempPivots, long size, int mergersPerTask, int mergerIdInTask, long taskSize) {
  const int WARPS = THREADS/W;
  int tid = threadIdx.x%W;
  int warpInBlock = threadIdx.x/W;
  int targetPivot = ((mergerIdInTask)*((taskSize)/mergersPerTask));
  T minVal,maxVal;
  int minIdx, maxIdx;

  __shared__ T candidates[K*WARPS];
  __shared__ int partitionVal[WARPS];

  if(threadIdx.x < WARPS) {
    partitionVal[threadIdx.x] = (size*K)/2;
  }

 volatile  __shared__ int startBoundary[K*WARPS];
 volatile  __shared__ int endBoundary[K*WARPS];

  // Initialize boundary positions
  if(threadIdx.x < K*WARPS) {
    startBoundary[threadIdx.x] = 0;
    endBoundary[threadIdx.x] = size-1;
    tempPivots[tid] = size/2;
  }

// first warp of task begins at start of every list
  if(mergerIdInTask == 0 && tid < K) {
    tempPivots[tid] = 0;
  }

__syncthreads();

// find min and max elts of list
  if(mergerIdInTask > 0) {
    // Set initial candidate values
    if(tid < K) {
      candidates[warpInBlock*K+tid] = data[size*tid + tempPivots[tid]];
    }

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
//          {
            partVal[warpInBlock] = candidates[warpInBlock*K + minIdx];
            partList[warpInBlock] = minIdx;
//          }
        } 
        else { // If we need to move max candidate
          partitionVal[warpInBlock] -= (tempPivots[maxIdx] - startBoundary[warpInBlock*K + maxIdx])/2; // Increase rank of current partition boundary
          endBoundary[warpInBlock*K + maxIdx] = tempPivots[maxIdx];
          tempPivots[maxIdx]=(endBoundary[warpInBlock*K+maxIdx]+startBoundary[warpInBlock*K+maxIdx])/2;
          candidates[warpInBlock*K + maxIdx] = data[size*maxIdx + tempPivots[maxIdx]];
          if(startBoundary[warpInBlock*K + maxIdx] >= endBoundary[warpInBlock*K + maxIdx] && tempPivots[maxIdx] > 0) 
 //         {
            partVal[warpInBlock] = candidates[warpInBlock*K + maxIdx];
            partList[warpInBlock] = maxIdx;
          }
        }
      }
      // Binary search each other list to find predecessor of partitioning value
      __syncthreads();

      int step;
      if(tid < K) {
        if(tid != partList[warpInBlock]) {
          tempPivots[tid] = size/2;
          step = size/4;

          while(step >= 1) {
            if(!f((data[size*tid + tempPivots[tid]]), partVal[warpInBlock])) 
              tempPivots[tid] -= step;
            else 
              tempPivots[tid] += step;
            step /=2;
          }
          if(tempPivots[tid] > 0 && cmp(partVal[warpInBlock], (data[size*tid + tempPivots[tid]-1])))
            tempPivots[tid]--;
          if(f((data[size*tid + tempPivots[tid]]), partVal[warpInBlock]))
            tempPivots[tid]++;
        }
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
__device__ void single_pivot_partition(T* data, int* tempPivots, long size, long mergersPerTask, long mergerIdInTask, long taskSize) {
  #ifdef PIPELINE
  int tid = threadIdx.x;
  #else
  int tid = threadIdx.x%W;
  #endif
  if (mergerIdInTask == 0) {
    if (tid < K) {
      tempPivots[tid] = 0;
    }
  } else {
    __shared__ T values[K];
    int order = tid;
    int L = (taskSize+size-1)/size;
    if (tid >= L && tid < K) {
      values[tid] = INT_MAX;
    } else if (tid<L) {
      T val = data[size*tid + ((taskSize-tid*size > size) ? size : taskSize-tid*size)/2];
      values[tid] = val;
    }

    int delta=1;
    for (int i=0; i<PL; i++) {
      delta*=2;
      int direction = ((order / delta) % 2 == 0);
      for (int j=delta; j>1; j /= 2) {
          // Direction indicates the direction in which it is sorted. 1=increasing, 0=decreasing
          // Higher indicates whether the particular thread should contain the higher or lower element.
          int higher = ((order % j) < j/2) ^ direction;
          // Exchange with the thread that has ID that is XOR with j/2 of mine. Since XOR is the same forwards as backwards, the opposite thread gets my ID, too.
          int partner = order ^ (j/2);
          __syncwarp();
          T myVal = values[order];
          __syncwarp();
          if (!equals<T,f>(myVal, values[partner]) && (!higher ^ f(myVal, values[partner]))) {
            values[partner] = myVal;
            order = partner;
          }
          __syncwarp();
      }
    }

    __syncwarp();

    if (tid<L) {
      unsigned int ballot = __ballot_sync(FULL_MASK, tid<L);
      Pair<T> myPair;
      int left;
      int right;
      int mid;
      int completed;
      __shared__ int totalCompleted;
      __shared__ int sum;
      int mySum;
      long target = mergerIdInTask*taskSize/mergersPerTask;
      taskSize--;

      totalCompleted=0;
      sum=0;
      left = -1;
      right = (taskSize-tid*size > size) ? size : taskSize-tid*size;
      mid = right/2;
      completed = 0;
      mySum = mid;

      // Use kogge-stone to get sum
      int shift=1;
      int neighbor;
      for (int i=0; i<PL; i++) {
        neighbor = __shfl_up_sync(ballot, mySum, shift);
        if (tid - shift >= 0) {
          mySum += neighbor;
        }
        shift *= 2;
      }
      if (tid == L-1) {
        sum = mySum;
      }
      __syncwarp(ballot);


      // moveMax is true if we move the max pivot. The thread that moves the pivot has active as 1 and inactive as 0.
      int moveMax;
      int active;
      __shared__ T temp;
      __shared__ int readjust;
      __shared__ int startOffset;
      __shared__ int endOffset;
      __shared__ unsigned int activeThreadBit;

      startOffset=0;
      endOffset=0;

      __syncwarp(ballot);

      while (totalCompleted < L-1) {
        moveMax = (sum >= target) ? 1 : 0;
        active = (moveMax && order == L-1-endOffset) || (!moveMax && order == startOffset);

        __syncwarp(ballot);

        if (active) {
          /*
          if (blockIdx.x == 1) {
            printf("\nmoveMax: %d, endOffset: %d, startOffset: %d\nactive thread: %d\n", moveMax, endOffset, startOffset, threadIdx.x);
            printf("Values: ");
            for (int i=startOffset; i<=L-1-endOffset; i++) {
              printf("%d ", values[i]);
            }
            printf("\n");
          }
          */
          if (moveMax) {
            right = mid;
          } else if (!moveMax) {
            left = mid;
          }
          sum -= mid;
          mid = (left+right)/2;
          sum += mid;
          if (left + 1 == right) {
            totalCompleted++;
            tempPivots[tid] = right;
            sum++;
            if (moveMax) {
              endOffset++;
            } else {
              startOffset++;
            }
            readjust = 0;
            return;
          }
          temp = data[size*tid + mid];
          readjust = 1;
          activeThreadBit = 2 << tid;
        }

        __syncwarp(ballot);

        int newOrder=order;
        if (readjust && !active) {
          if (moveMax) {
            T temporaryReadVal = values[order];
            // unsigned mask = ballot ^ activeThreadBit;
            __syncwarp(ballot & !activeThreadBit);
            // if (blockIdx.x == 1) printf("|");
            if (!equals<T,f>(temp, temporaryReadVal) && f(temp, temporaryReadVal)) {
              newOrder = order+1;
              values[newOrder] = temporaryReadVal;
            }
          } else {
            T temporaryReadVal = values[order];
            // unsigned mask = ballot ^ activeThreadBit;
            __syncwarp(ballot & !activeThreadBit);
            // if (blockIdx.x == 1) printf("|");
            if (!equals<T,f>(temp, temporaryReadVal) && f(temporaryReadVal, temp)) {
              newOrder = order-1;
              values[newOrder] = temporaryReadVal;
            }
          }
        }

        __syncwarp(ballot);
        
        int position = __popc(__ballot_sync(FULL_MASK, (!active && order != newOrder))); // Computes which threads have changed order. If 
        // printf("position: %d\n", position);
        if (readjust && active) {
          if (moveMax) {
            newOrder = L-1-endOffset-position;
          } else {
            newOrder=startOffset+position;
          }
          values[newOrder] = temp;
        }
        

        __syncwarp(ballot);
        order = newOrder;
      }

      mid += target-sum;
      tempPivots[tid] = mid;
    } else if (tid < K) {
      tempPivots[tid] = 0;
    }
  }
}

template<typename T, fptr_t f>
__device__ void double_pivot_partition(T* data, int* tempPivots, long size, long mergersPerTask, long mergerIdInTask, long taskSize) {
  #ifdef PIPELINE
  int tid = threadIdx.x;
  #else
  int tid = threadIdx.x%W;
  #endif
  if (mergerIdInTask == 0) {
    if (tid < K) {
      tempPivots[tid] = 0;
    }
  } else if (tid == 0) {
    long left[K];
    long right[K];
    long mid[K];
    int completed[K];
    int totalCompleted=0;
    int L = (taskSize+size-1)/size;
    if (L == 1) {
      tempPivots[0] = ((mergerIdInTask)*(taskSize/mergersPerTask));
      for (int i=1; i<K; i++) {
        tempPivots[i] = 0;
      }
      return;
    }
    for (int i=0; i<L; i++) {
      left[i] = -1;
      if (taskSize > 0) {
        right[i] = (taskSize > size) ? size : taskSize;
      } else {
        right[i] = 0;
      }
      mid[i] = right[i]*mergerIdInTask / mergersPerTask;
      taskSize -= size;
      completed[i] = 0;
    }

    while (totalCompleted < L-1) {
      int firstIndex;
      for (int i=L-1; i>=0; i--) {
        if (!completed[i]) {
          firstIndex = i;
        }
      }
      T max = data[size*firstIndex + mid[firstIndex]];
      T min = data[size*firstIndex + mid[firstIndex]];
      int maxIdx = firstIndex;
      int minIdx = firstIndex;
      int minIdxShift;
      int maxIdxShift;
      int shift;
      for (int i=1; i<L; i++) {
        if (!completed[i]) {
          if (equals<T,f>(max, data[size*i + mid[i]]) || f(max, data[size*i + mid[i]])) {
            maxIdx = i;
            max = data[size*i + mid[i]];
          }
          if (f(data[size*i + mid[i]], min) && !equals<T,f>(min, data[size*i + mid[i]])) {
            minIdx = i;
            min = data[size*i + mid[i]];
          }
        }
      }
      
      right[maxIdx] = mid[maxIdx];
      left[minIdx] = mid[minIdx];

      maxIdxShift = right[maxIdx] - (left[maxIdx]+right[maxIdx])/2;
      minIdxShift = (left[minIdx]+right[minIdx])/2 - left[minIdx];

      if (minIdxShift < maxIdxShift) {
        shift = minIdxShift;
      } else {
        shift = maxIdxShift;
      }

      mid[maxIdx] = right[maxIdx] - shift;
      mid[minIdx] = left[minIdx] + shift;
      
      if (left[maxIdx] + 1 == right[maxIdx]) {
        totalCompleted++;
        completed[maxIdx] = 1;
      }
      
      if (left[minIdx] + 1 == right[minIdx]) {
        totalCompleted++;
        completed[minIdx] = 1;
      }
    }

    for (int i=0; i<K; i++) {
      if (i<L) {
        if (!completed[i]) {
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

// If there is an edge case
template<typename T, fptr_t f>
__global__ void findPartitions(T* data, int* pivots, int size, int tasks, int edgeCaseTaskSize) {
  #ifdef PIPELINE
  __shared__ int myPivots[K];
  int mergerIdx = blockIdx.x;
  int mergersPerTask = gridDim.x/tasks;
  int myTask = blockIdx.x / mergersPerTask;
  int mergerIdInTask = blockIdx.x - myTask*mergersPerTask;
  int tid = threadIdx.x;
  #else
  __shared__ int myPivotsRaw[K*(THREADS/W)];
  int* myPivots = myPivotsRaw+(threadIdx.x/W)*K;
  int mergerIdx = blockIdx.x*(THREADS/W) + (threadIdx.x/W);
  int mergersPerTask = (gridDim.x*(THREADS/W))/tasks;
  int myTask = mergerIdx / mergersPerTask;
  int mergerIdInTask = mergerIdx - myTask*mergersPerTask;
  int tid = threadIdx.x%W;
  #endif
  int taskOffset = myTask*size*K;
  int taskSize = (myTask<tasks-1) ? K*size : edgeCaseTaskSize;

  if(myTask < tasks) {
    // In this case, we don't have the edge case, so we reuse the same warp_partition function
    #ifdef PIVOT_MOVES
    #if (PIVOT_MOVES == 1)
    single_pivot_partition<T, f>(data+taskOffset, myPivots, size, mergersPerTask, mergerIdInTask, taskSize);
    #else
    double_pivot_partition<T, f>(data+taskOffset, myPivots, size, mergersPerTask, mergerIdInTask, taskSize);
    #endif
    #else
    old_partition<T, f>(data+taskOffset, myPivots, size, mergersPerTask, mergerIdInTask, taskSize);
    #endif
  }

  if(tid < K) {
    pivots[mergerIdx*K+tid] = myPivots[tid];
  }

  if(myTask==tasks-1 && tid<K) {  // Fill last K spots with end values
    int difference = edgeCaseTaskSize - tid * size;
    difference = (difference > 0) * difference;
    int end = (difference < size)*difference + (size <= difference)*size; // taking MIN of difference and size using predicates
    pivots[mergersPerTask*tasks*K+tid] = end;
  }
}

void __global__ printPartitions(int* pivots, int mergers) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("-----------------------------------------------------------------------\n");
    for (int i=0; i<K*mergers+K; i++) {
      printf("%d ", pivots[i]);
    }
    printf("\n");
  }
  
}
