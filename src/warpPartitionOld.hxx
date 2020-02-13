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

#ifndef warpPartition_hxx
	#define warpPartition_hxx
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include "params.h"


// SAVING THIS FILE 2/11/20 AS A BACKUP FOR WHEN I MAKE CHANGES


/*	Parameters
 *	warpIdInTask = warpIdx - myTask*warpsPerTask;
 */

// Find a set of pivots for a given partition
template<typename T>
__device__ void warpPartition(T* data, int* tempPivots, int size, int warpsPerTask, int warpIdInTask) {
	const int WARPS = THREADS/W; // this becomes 1?
	int tid = threadIdx.x%W; // this is the same as threadIdx.x when W is 32
	int warpInBlock = threadIdx.x/W; // always 0? b/c threadIdx.x is between 0 and 31?
	int targetPivot = ((warpIdInTask)*((size*K)/warpsPerTask)); // 
	T minVal,maxVal;
	int minIdx, maxIdx;

	__shared__ T candidates[K*WARPS]; // K = 4 candidates
	__shared__ int partitionVal[WARPS]; // just 1 value

	if (threadIdx.x == 0) {
		printf("targetPivot: %d\n", targetPivot);
	}

	if(threadIdx.x < WARPS) { // aka threadIdx.x < 1
		partitionVal[threadIdx.x] = (size*K)/2; // sets partitionVal[0] to 2048 for the 4096 case .. halfway point?
		printf("set partitionVal[%d] = %d\tsize = %d\tK = %d\n", threadIdx.x, (size*K)/2, size, K);
	}

	volatile  __shared__ int startBoundary[K*WARPS];
	volatile  __shared__ int endBoundary[K*WARPS];

	// Initialize boundary positions
	if(threadIdx.x < K*WARPS) { // aka threadIdx.x < K = 4
		startBoundary[threadIdx.x] = 0;
		endBoundary[threadIdx.x] = size-1;
		tempPivots[tid] = size/2;
		// printf("tempPivots[%d] = %d\n", tid, tempPivots[tid]);
	}

	// first warp of task begins at start of every list
	if(warpIdInTask == 0 && tid < K) {
		tempPivots[tid] = 0;
		// printf("tempPivots[%d] = 0\n", tid);
	}

	__syncthreads();

	// find min and max elts of list
	if(warpIdInTask > 0) {


		// Set initial candidate values
		if(tid < K) { // only uses the K lowest threads.. why?
			// printf("warpPartition blockIdx: %d\t threadIdx: %d\n", blockIdx.x, threadIdx.x);
			candidates[warpInBlock*K+tid] = data[size*tid + tempPivots[tid]]; // warpInBlock * K + tid = 0 * K + tid = tid, so candidates[tid]..?
		}

		__shared__ T partVal[THREADS/W]; // this and the next array are size 1
		__shared__ int partList[THREADS/W];
		// Sequential section per warp 
		if(tid==0) {
			partVal[warpInBlock]=MAXVAL;
			int tempPartitionVal;
			int iterIdx;
			minVal=0; // I think minVal and maxVal are set to arbitrary values since the next while loop always happens
			maxVal=1; // MIGHT DELETE THESE TWO LINES

			while(partVal[warpInBlock] == MAXVAL) {
			minVal=MAXVAL; // max and min 32 bit values defined in cmp.hxx (this is in int, so doesn't really match with template type T)
			maxVal=MINVAL;
				// find min and max - OPTIMIZE use K threads to do min and max reduction
				for(int i=0; i<K; i++) {
					iterIdx = warpInBlock*K + i; // 0 * K + i = i
					// compares candidate values with minVal and maxVal to properly set them, also keeps the index (minIdx, maxIdx)
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
				// printf("partitionVal[%d]: %d\n", warpInBlock, tempPartitionVal);

				// If we need to move min candidate
				if(targetPivot >= tempPartitionVal) {
					partitionVal[warpInBlock] += (endBoundary[warpInBlock*K + minIdx] - tempPivots[minIdx])/2; // Increase rank of current partition boundary
					startBoundary[warpInBlock*K + minIdx] = tempPivots[minIdx];
					tempPivots[minIdx]=(endBoundary[warpInBlock*K+minIdx]+startBoundary[warpInBlock*K+minIdx])/2;
					// printf("tempPivots[%d] = %d\n", minIdx, tempPivots[minIdx]);
					if(tempPivots[minIdx] == startBoundary[warpInBlock*K + minIdx]) { // Edge case
						tempPivots[minIdx]++;
						// printf("tempPivots[%d] = %d\n", minIdx, tempPivots[minIdx]);
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
					// printf("tempPivots[%d] = %d\n", maxIdx, tempPivots[maxIdx]);
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
				// printf("tempPivots[%d] = %d\n", tid, tempPivots[tid]);
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

// Find pivots K pivots for each warp within a 'task' (a group of K lists)
// Pivots define the start of the partition that each warp will work on merging
template<typename T>
__global__ void findPartitions(T* data, T*output, int* pivots, int size, int numLists, int tasks, int P) {
	__shared__ int myPivotsRaw[K*(THREADS/W)]; // size K when THREADS = W
	int warpInBlock = threadIdx.x/W;
	int* myPivots = myPivotsRaw+(warpInBlock*K);
	int warpIdx = (blockIdx.x)*(THREADS/W) + warpInBlock;
	int tid = threadIdx.x%W;
	int warpsPerTask;
	int myTask;
	int taskOffset;
	int warpIdInTask;
	int totalWarps = P*(THREADS/W);

	// printf("FindPartition blockIdx: %d\t threadIdx: %d\n", blockIdx.x, threadIdx.x);

	warpsPerTask = totalWarps/tasks; // floor
	if(warpsPerTask <= 1) {
		// printf("warpsPerTasks LESS than 1\ttotalWarps: %d\ttasks: %d\n", totalWarps, tasks);
		if(tid < K) {
			pivots[warpIdx*K+tid] = 0;
		}
		if(blockIdx.x==0 && threadIdx.x < K) {
			pivots[totalWarps*K+threadIdx.x] = size;
		}
	}
	else {
		// printf("warpsPerTasks GREATER than 1\ttotalWarps: %d\ttasks: %d\n", totalWarps, tasks);
		myTask = warpIdx / warpsPerTask; // If we have extra warps, just have them do no work...
		if(myTask < tasks) {
			taskOffset = myTask*size*K;
			warpIdInTask = warpIdx - myTask*warpsPerTask;

			warpPartition<T>(data+taskOffset, myPivots, size, warpsPerTask, warpIdInTask);

			if(tid < K) {
				pivots[warpIdx*K+tid] = myPivots[tid];
			}
			if(blockIdx.x ==0 && threadIdx.x<K) {  // Fill last K spots with max val
				pivots[totalWarps*K+threadIdx.x] = size;
			}
		}
	}
	for (int i = 0; i < (P + 1) * K; i++) {
		if (threadIdx.x == 31 && blockIdx.x == 127) {
			printf("BlockIdx: %d\t threadIdx: %d\tpivots[%d]: %d\n", blockIdx.x, threadIdx.x, i, pivots[i]);
		}
	}
}

/* FOR DEBUGGING - MAKES SURE PIVOTS MAKE A VALID PARTITION */
template<typename T>
void __global__ testPartitioning(T* data, int* pivots, int size, int tasks, int P) {
	int warpsPerTask;
	int totalWarps = P*(THREADS/W);
	warpsPerTask = totalWarps/tasks; // floor
	// printf("Made it in testPartitioning\n");
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
		}
		if(error)
			printf("Partitioning failed\n");
		else
			printf("Partitioning correct!\n");
	}
	// printf("Finished testing partitioning.\n");
}

#endif