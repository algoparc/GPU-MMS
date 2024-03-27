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
  } else if (tid == 0) {
    long left[K];
    long right[K];
    long mid[K];
    int completed[K];
    T PQMaxVals[K];
    int PQMaxIndices[K];
    T PQMinVals[K];
    int PQMinIndices[K];
    int totalCompleted=0;
    int L = (taskSize+size-1)/size;
    int sum=0;
    long target = mergerIdInTask*taskSize/mergersPerTask;
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
      mid[i] = right[i]/2;
      sum += mid[i];
      taskSize -= size;
      completed[i] = 0;
      PQMaxVals[i] = data[size*i + mid[i]];
      PQMaxIndices[i] = i;
      PQMinVals[i] = data[size*i + mid[i]];
      PQMinIndices[i] = i;
    }

    //Build max-heap
    T temp;
    int idxTemp;
    for (int i=(L-2)/2; i>=0; i--) {
      //Max-Heapify
      int j=i;
      int done=0;
      while (2*j+1 < L && !done) {
        if (2*j+2 < L) {
          if (f(PQMaxVals[2*j+1], PQMaxVals[2*j+2])) {
            if (f(PQMaxVals[j], PQMaxVals[2*j+2])) {
              temp = PQMaxVals[j];
              PQMaxVals[j] = PQMaxVals[2*j+2];
              PQMaxVals[2*j+2] = temp;

              idxTemp = PQMaxIndices[j];
              PQMaxIndices[j] = PQMaxIndices[2*j+2];
              PQMaxIndices[2*j+2] = idxTemp;
              j = 2*j+2;
            } else {
              done = 1;
            }
          } else {
            if (f(PQMaxVals[j], PQMaxVals[2*j+1])) {
              temp = PQMaxVals[i];
              PQMaxVals[j] = PQMaxVals[2*j+1];
              PQMaxVals[2*j+1] = temp;

              idxTemp = PQMaxIndices[j];
              PQMaxIndices[j] = PQMaxIndices[2*j+1];
              PQMaxIndices[2*j+1] = idxTemp;
              j = 2*j+1;
            } else {
              done = 1;
            }
          }
        } else {
          if (f(PQMaxVals[j], PQMaxVals[2*j+2])) {
              temp = PQMaxVals[j];
              PQMaxVals[j] = PQMaxVals[2*j+2];
              PQMaxVals[2*j+2] = temp;

              idxTemp = PQMaxIndices[j];
              PQMaxIndices[j] = PQMaxIndices[2*j+2];
              PQMaxIndices[2*j+2] = idxTemp;
              j = 2*j+2;
            } else {
              done = 1;
            }
        }
      }
    }

    // Build min-heap
    for (int i=(L-2)/2; i>=0; i--) {
      //Min-Heapify
      int j=i;
      int done=0;
      while (2*j+1 < L && !done) {
        if (2*j+2 < L) {
          if (f(PQMinVals[2*j+2], PQMinVals[2*j+1])) {
            if (f(PQMinVals[2*j+2], PQMinVals[j])) {
              temp = PQMinVals[j];
              PQMinVals[j] = PQMinVals[2*j+2];
              PQMinVals[2*j+2] = temp;

              idxTemp = PQMinIndices[j];
              PQMinIndices[j] = PQMinIndices[2*j+2];
              PQMinIndices[2*j+2] = idxTemp;
              j = 2*j+2;
            } else {
              done = 1;
            }
          } else {
            if (f(PQMinVals[2*j+1], PQMinVals[j])) {
              temp = PQMinVals[j];
              PQMinVals[j] = PQMinVals[2*j+1];
              PQMinVals[2*j+1] = temp;

              idxTemp = PQMinIndices[j];
              PQMinIndices[j] = PQMinIndices[2*j+1];
              PQMinIndices[2*j+1] = idxTemp;
              j = 2*j+1;
            } else {
              done = 1;
            }
          }
        } else {
          if (f(PQMinVals[2*j+1], PQMinVals[j])) {
            temp = PQMinVals[j];
            PQMinVals[j] = PQMinVals[2*j+1];
            PQMinVals[2*j+1] = temp;

            idxTemp = PQMinIndices[j];
            PQMinIndices[j] = PQMinIndices[2*j+1];
            PQMinIndices[2*j+1] = idxTemp;
            j = 2*j+1;
          } else {
            done = 1;
          } 
        }
      }
    }


    int idx=0;

    while (totalCompleted < L-1) {
      int moveMax = sum >= target;
      
      if (moveMax) {
        idx = PQMaxIndices[0];
        right[idx] = mid[idx];
      } else {
        idx = PQMinIndices[0];
        left[idx] = mid[idx];
      }
      sum -= mid[idx];
      mid[idx] = (left[idx]+right[idx])/2;
      sum += mid[idx];
      
      if (left[idx] + 1 == right[idx]) {
        totalCompleted++;
        completed[idx] = 1;
        sum++; // Add 1 to sum, because mid currently equals left, and we return right[];
        if (moveMax) {
          PQMaxIndices[0] = PQMaxIndices[L-totalCompleted];
          PQMaxVals[0] = PQMaxVals[L-totalCompleted];
        } else {
          PQMinIndices[0] = PQMinIndices[L-totalCompleted];
          PQMinVals[0] = PQMinVals[L-totalCompleted];
        }
      } else {
        if (moveMax) {
          PQMaxVals[0] = data[size*idx + mid[idx]];
          int done = 0;
          int j=0;
          while (2*j+1 < L-totalCompleted && !done) {
            if (2*j+2 < L-totalCompleted) {
              if (f(PQMaxVals[2*j+1], PQMaxVals[2*j+2])) {
                if (f(PQMaxVals[j], PQMaxVals[2*j+2])) {
                  temp = PQMaxVals[j];
                  PQMaxVals[j] = PQMaxVals[2*j+2];
                  PQMaxVals[2*j+2] = temp;

                  idxTemp = PQMaxIndices[j];
                  PQMaxIndices[j] = PQMaxIndices[2*j+2];
                  PQMaxIndices[2*j+2] = idxTemp;
                  j = 2*j+2;
                } else {
                  done = 1;
                }
              } else {
                if (f(PQMaxVals[j], PQMaxVals[2*j+1])) {
                  temp = PQMaxVals[j];
                  PQMaxVals[j] = PQMaxVals[2*j+1];
                  PQMaxVals[2*j+1] = temp;

                  idxTemp = PQMaxIndices[j];
                  PQMaxIndices[j] = PQMaxIndices[2*j+1];
                  PQMaxIndices[2*j+1] = idxTemp;
                  j = 2*j+1;
                } else {
                  done = 1;
                }
              }
            }
          }
        } else {
          PQMinVals[0] = data[size*idx + mid[idx]];
          int done = 0;
          int j=0;
          while (2*j+1 < L-totalCompleted && !done) {
            if (2*j+2 < L-totalCompleted) {
              if (f(PQMinVals[2*j+2], PQMinVals[2*j+1])) {
                if (f(PQMinVals[2*j+2], PQMinVals[j])) {
                  temp = PQMinVals[j];
                  PQMinVals[j] = PQMinVals[2*j+2];
                  PQMinVals[2*j+2] = temp;

                  idxTemp = PQMinIndices[j];
                  PQMinIndices[j] = PQMinIndices[2*j+2];
                  PQMinIndices[2*j+2] = idxTemp;
                  j = 2*j+2;
                } else {
                  done = 1;
                }
              } else {
                if (f(PQMinVals[2*j+1], PQMinVals[j])) {
                  temp = PQMinVals[j];
                  PQMinVals[j] = PQMinVals[2*j+1];
                  PQMinVals[2*j+1] = temp;

                  idxTemp = PQMinIndices[j];
                  PQMinIndices[j] = PQMinIndices[2*j+1];
                  PQMinIndices[2*j+1] = idxTemp;
                  j = 2*j+1;
                } else {
                  done = 1;
                }
              }
            }
          }
        }
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
    single_pivot_partition<T, f>(data+taskOffset, myPivots, size, mergersPerTask, mergerIdInTask, taskSize);
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

void __global__ printPartitions(int* pivots, int blocks) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("-----------------------------------------------------------------------\n");
    for (int i=0; i<K*blocks+K; i++) {
      printf("%d ", pivots[i]);
    }
    printf("\n");
  }
  
}
