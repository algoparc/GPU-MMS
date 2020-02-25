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





/*	warpPartition finds the pivots for a small section of the data.
 *  T - the data type of the values in the arrays
 *	data - an array of unsorted values (on the GPU)
 *	tempPivots - an array of points in the data that dictate where the merging happens	
 *	size - the size of the arrays that are being merged at this level of the mergesort
 *	warpsPerTask - 
 *	warpIdInTask - 
 */

// Find a set of pivots for a given partition
template<typename T>
__device__ void warpPartition(T* data, int* tempPivots, int size, int warpsPerTask, int warpIdInTask) {
	#ifdef SKIP_PADDED_PARTITION
	// If a padded section is reach, skip this step since all of the values in this section are the same.	
	if (data[0] == RANGE) {
		// printf("RANGE VALUE FOUND AT START OF PARTITION\n");
		// Still set evenly distributed pivots just in case..
		for (int i = 0; i < K; i++) {
			tempPivots[i] = i * (size/K);
		}
		return;
	}
	#endif

	const int WARPS = THREADS/W; // this becomes 1?
	int tid = threadIdx.x%W; // this is the same as threadIdx.x when W is 32
	int warpInBlock = threadIdx.x/W; // always 0? b/c threadIdx.x is between 0 and 31?
	int targetPivot = ((warpIdInTask)*((size*K)/warpsPerTask)); // 
	T minVal,maxVal;
	int minIdx, maxIdx;

	__shared__ T candidates[K*WARPS]; // K = 4 candidates
	__shared__ int partitionVal[WARPS]; // just 1 value


	if(threadIdx.x < WARPS) { // aka threadIdx.x < 1
		partitionVal[threadIdx.x] = (size*K)/2; // sets partitionVal[0] to 2048 for the 4096 case .. halfway point?
	}

	volatile  __shared__ int startBoundary[K*WARPS];
	volatile  __shared__ int endBoundary[K*WARPS];

	// Initialize boundary positions
	if(threadIdx.x < K*WARPS) { // aka threadIdx.x < K = 4
		startBoundary[threadIdx.x] = 0;
		endBoundary[threadIdx.x] = size-1;
		tempPivots[tid] = size/2;
	}

	// first warp of task begins at start of every array
	if(warpIdInTask == 0 && tid < K) {
		tempPivots[tid] = 0;
	}

	__syncthreads();

	// find min and max elts of array
	if(warpIdInTask > 0) {


		// Set initial candidate values
		if(tid < K) { // only uses the K lowest threads.. why?
			candidates[warpInBlock*K+tid] = data[size*tid + tempPivots[tid]]; // warpInBlock * K + tid = 0 * K + tid = tid, so candidates[tid]..?
		}

		__shared__ T partVal[THREADS/W]; // this and the next array are size 1
		__shared__ int partArray[THREADS/W];
		// Sequential section per warp 
		if(tid==0) {
			partVal[warpInBlock]=MAXVAL;
			int tempPartitionVal;
			int iterIdx;

			while(partVal[warpInBlock] == MAXVAL) {
				// Assign initial placeholder values for minVal and maxVal 
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
					if(startBoundary[warpInBlock*K + minIdx] >= endBoundary[warpInBlock*K + minIdx] && tempPivots[minIdx] < size-1) {
						partVal[warpInBlock] = candidates[warpInBlock*K + minIdx];
					}
					partArray[warpInBlock] = minIdx;
				} 
				else { // If we need to move max candidate
					partitionVal[warpInBlock] -= (tempPivots[maxIdx] - startBoundary[warpInBlock*K + maxIdx])/2; // Increase rank of current partition boundary
					endBoundary[warpInBlock*K + maxIdx] = tempPivots[maxIdx];
					tempPivots[maxIdx]=(endBoundary[warpInBlock*K+maxIdx]+startBoundary[warpInBlock*K+maxIdx])/2;
					candidates[warpInBlock*K + maxIdx] = data[size*maxIdx + tempPivots[maxIdx]];
					if(startBoundary[warpInBlock*K + maxIdx] >= endBoundary[warpInBlock*K + maxIdx] && tempPivots[maxIdx] > 0) {
						partVal[warpInBlock] = candidates[warpInBlock*K + maxIdx];
					}
					partArray[warpInBlock] = maxIdx;
				}
			}
		}
		__syncthreads();

		// Binary search each other array to find predecessor of partitioning value
		int step;
		if(tid < K) {
			if(tid != partArray[warpInBlock]) {
				tempPivots[tid] = size/2; // Start with the halfway point
				step = size/4;

				while(step >= 1) {
					if(!cmp((data[size*tid + tempPivots[tid]]), partVal[warpInBlock])) 
						tempPivots[tid] -= step;
					else 
						tempPivots[tid] += step;
					step /=2; // Halve the step size after each iteration
				}
				// Update the pivots
				if(tempPivots[tid] > 0 && cmp(partVal[warpInBlock], (data[size*tid + tempPivots[tid]-1])))
					tempPivots[tid]--;
				if(cmp((data[size*tid + tempPivots[tid]]), partVal[warpInBlock]))
					tempPivots[tid]++;
			}
		}
	}
}

/*	Find pivots K pivots for each warp within a 'task' (a group of K arrays).
 *	Pivots define the start of the partition that each warp will work on merging.
 *  T - the data type of the values in the arrays
 *	data - an array of unsorted values (on the GPU)
 *	output - an array that will contain the rearranged values from input after merging (on the GPU)
 *	pivots - an array of points in the data that dictate where the merging happens	
 *	size - the size of the arrays that are being merged at this level of the mergesort
 *	tasks - the number of merges happening at this current level of the mergesort
 *	P - the number of blocks used on the GPU
 */

template<typename T>
__global__ void findPartitions(T* data, T*output, int* pivots, int size, int tasks, int P) {
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

	#ifdef PRINT_DEBUG
	printf("FindPartition blockIdx: %d\t threadIdx: %d\n", blockIdx.x, threadIdx.x);
	#endif

	warpsPerTask = totalWarps/tasks; // floor
	if(warpsPerTask <= 1) {
		#ifdef PRINT_DEBUG
		printf("warpsPerTasks LESS than 1\ttotalWarps: %d\ttasks: %d\n", totalWarps, tasks);
		#endif

		if(tid < K) {
			pivots[warpIdx*K+tid] = 0;
		}
		if(blockIdx.x==0 && threadIdx.x < K) {
			pivots[totalWarps*K+threadIdx.x] = size;
		}
	}
	else {
		#ifdef PRINT_DEBUG
		printf("warpsPerTasks GREATER than 1\ttotalWarps: %d\ttasks: %d\n", totalWarps, tasks);
		#endif

		myTask = warpIdx / warpsPerTask; // If we have extra warps, just have them do no work...
		
		#ifdef PRINT_DEBUG
		if (myTask >= tasks) { // why do we just ignore the extra warps?
			printf("myTask: %d\ttasks: %d\n", myTask, tasks);
		}
		#endif

		if(myTask < tasks) { // myTask becomes 0 for 4096 (maybe for other nice cases, too)
			taskOffset = myTask*size*(K); // How does changing this change the size of the partitions
			warpIdInTask = warpIdx - myTask*warpsPerTask;
			
			#ifdef PRINT_DEBUG
			if (size == 1024) {
				printf("taskOffset: %d\tmyTask: %d\ttasks: %d\tsize: %d\tK: %d\n", taskOffset, myTask, tasks, size, K);
			}
			#endif

			warpPartition<T>(data+taskOffset, myPivots, size, warpsPerTask, warpIdInTask);

			if(tid < K) {
				pivots[warpIdx*K+tid] = myPivots[tid];
			}
			if(blockIdx.x ==0 && threadIdx.x<K) {  // Fill last K spots with max val
				pivots[totalWarps*K+threadIdx.x] = size;
			}
		}
	}
}

/* FOR DEBUGGING - MAKES SURE PIVOTS MAKE A VALID PARTITION */
template<typename T>
void __global__ testPartitioning(T* data, int* pivots, int size, int tasks, int P) {
	int warpsPerTask;
	int totalWarps = P*(THREADS/W);
	warpsPerTask = totalWarps/tasks; // floor
	#ifdef PRINT_DEBUG
	printf("Made it in testPartitioning\n");
	#endif
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
		// Print results of this test only if necessary (failure).
		if(error) {
			printf("Partitioning FAILED\n");
		}
		else {
			#ifdef PRINT_DEBUG
			printf("Partitioning SUCCEEDED\n");
			#endif
		}
	}
	#ifdef PRINT_DEBUG
	printf("Finished testing partitioning.\n");
	#endif
}

#endif