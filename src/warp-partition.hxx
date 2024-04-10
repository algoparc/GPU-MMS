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

template<typename T>
struct pair {
  T val;
  int index;

  pair(T v, int id) {
    val = v;
    index = id;
  }
};

__forceinline__ __device__ int parent(int index) {
  return (index-1)/2;
}

__forceinline__ __device__ int leftChild(int index) {
  return 2*index+1;
}

__forceinline__ __device__ int rightChild(int index) {
  return 2*index+2;
}

template<typename T, fptr_t f>
__device__ int equals(T a, T b) {
  return !(f(a, b) ^ f(b, a));
}

template<typename T, fptr_t f>
__device__ void heapifyUp(T* heap, int* heapIndices, int size, int index) {
  T valSwap;
  int indexSwap;
  while (index>0 && f(heap[index], heap[parent(index)])) {
    valSwap = heap[index];
    heap[index] = heap[parent(index)];
    heap[parent(index)] = heap[index];
    
    indexSwap = heapIndices[index];
    heapIndices[index] = heapIndices[parent(index)];
    heapIndices[parent(index)] = heapIndices[index];
  }
}

template<typename T, fptr_t f>
__device__ void heapifyDown(T* heap, int* heapIndices, int size, int index) {
  T valSwap;
  int indexSwap;

  int done=0;
  while (index<size && !done) {
    if (rightChild(index) >= size) {
      if (leftChild(index) >= size) {
        done = 1;
      } else if (f(heap[index], heap[leftChild(index)])) {
        done=1;
      } else {
        valSwap = heap[index];
        heap[index] = heap[leftChild(index)];
        heap[leftChild(index)] = heap[index];
        
        indexSwap = heapIndices[index];
        heapIndices[index] = heapIndices[leftChild(index)];
        heapIndices[leftChild(index)] = heapIndices[index];

        index = leftChild(index);
      }
    } else if (f(heap[leftChild(index)], heap[rightChild(index)])) {
      if (f(heap[index], heap[leftChild(index)])) {
        done=1;
      } else {
        valSwap = heap[index];
        heap[index] = heap[leftChild(index)];
        heap[leftChild(index)] = heap[index];
        
        indexSwap = heapIndices[index];
        heapIndices[index] = heapIndices[leftChild(index)];
        heapIndices[leftChild(index)] = heapIndices[index];

        index = leftChild(index);
      }
    } else {
      if (f(heap[index], heap[rightChild(index)])) {
        done=1;
      } else {
        valSwap = heap[index];
        heap[index] = heap[rightChild(index)];
        heap[rightChild(index)] = heap[index];
        
        indexSwap = heapIndices[index];
        heapIndices[index] = heapIndices[rightChild(index)];
        heapIndices[rightChild(index)] = heapIndices[index];

        index = rightChild(index);
      }
    }
  }
}

template<typename T, fptr_t f>
__forceinline__ __device__ void push(T* minHeap, int* heapIndices, int size, T value, int index) {
  minHeap[size] = value;
  heapIndices[size] = index;
  heapifyUp<T,f>(size);
  size++;
}

template<typename T, fptr_t f>
pair<T> extractMin(T* minHeap, int* heapIndices, int size) {
  T value = minHeap[0];
  int intValue = heapIndices[0];
  pair<T> returnVal(value, intValue);
  size--;
  minHeap[0] = minHeap[size];
  heapIndices[0] = heapIndices[size];
  heapifyDown<T,f>(0);
  // Calculate or obtain values for 'value' and 'intValue'
  return returnVal;
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
    int left[K];
    int right[K];
    int mid[K];
    int completed[K];
    int totalCompleted=0;
    int L = (taskSize+size-1)/size;
    int sum=0;
    long target = mergerIdInTask*taskSize/mergersPerTask;
    taskSize--;
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

void __global__ printPartitions(int* pivots, int mergers) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("-----------------------------------------------------------------------\n");
    for (int i=0; i<K*mergers+K; i++) {
      printf("%d ", pivots[i]);
    }
    printf("\n");
  }
  
}
