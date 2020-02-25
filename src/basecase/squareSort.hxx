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

#ifndef squareSort_hxx
	#define squareSort_hxx
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
// #include "../params.h"
#include "sortRowMajor.hxx"

// Used for testing different size base cases
#define SQUARES 1
#define COLS 32

// Print column-major
__device__ void printSmem(int* smem) {

	if(threadIdx.x==0) {
		for(int j=0; j<(W); j++) {
			for(int i=0; i<(COLS)*SQUARES; i++) {
				printf("%d ", smem[j+i*(W)]);
			}
			printf("\n");
		}
		printf("\n");
	}
	__syncthreads();
}

void printResult(int* data) {
	for(int j=0; j<W; j++) {
		for(int i=0; i<ELTS*SQUARES; i++) {
			printf("%d ", data[j*ELTS*SQUARES+i]);
		}
	printf("\n");
	}
}

/*	Main basecase sorting Kernel. The unsorted array is divided into
 *	sections of size M, and each section is sorted.
 *	T - the data type of the values in the arrays
 *	f - a function pointer for the function used to compare values in the arrays
 *	data - the array of unsorted data
 *	N - the size of the array
 */
template<typename T, fptr_t f>
__global__ void squareSort(T* data, int N) {
	// Declare an array of elements for a single register.
	T regs[ELTS];

	// Calculate blockOffset, which is used for indexing on the global data
	int blockOffset = (N/gridDim.x)*blockIdx.x;
	// printf("%d\t%d\n", blockOffset, blockIdx.x);


	// N/gridDim.x always comes out to 1024?, sec always increments by M
	for(int sec = 0; sec < (N/gridDim.x); sec += M*(THREADS/W)) {
		// Load a section of the global data to be sorted on the registers
		for(int i=0; i<ELTS; i++) {
			// printf("%d\n", blockOffset + sec + (i*THREADS) + threadIdx.x);
			if ((blockOffset + sec + (i*THREADS) + threadIdx.x) >= N) { // edge case to avoid segmentation faults
				regs[i] = RANGE;
			}
			else {
				regs[i] = data[blockOffset + sec + (i*THREADS) + threadIdx.x];
			}
		}
		
		// Calculate more values that are used for indexing
		int tid = threadIdx.x%W;
		int warpId = threadIdx.x/W;
		int warpOffset = warpId*ELTS*W;
		__shared__ T sData[M];

		// Code that is needed for base case larger than 1024.
		// TODO: generalize for either 1024, 2048, or 4096 base case without changing code...
		/*
		sortSquareRowMajor(regs, warpId%2);
		warpSwap2(regs, sData, (warpId<(warpId^2))); 
		warpSwap4(regs, sData); 

		transposeSquares(regs, sData+warpOffset);

		for(int i=0; i<ELTS; i++) {
		data[blockOffset + sec + warpOffset + W*i + tid] = sData[threadIdx.x*W + (tid+i)%W];
		}
		*/

		// Sort the basecase section by using registers in a 32x32 pattern
		// if (regs[0] != RANGE) {
			sortSquareRowMajor<T,f>(regs, false);
		// }

		// Warps within a block use the same shared memory, so they have to take turns transposing
		// This lets us have more warps and increases performance!
		for(int i=0; i<warpId; i++) {
			__syncthreads();
		}

		// Transpose in shared memory then write to global
		transposeSquares<T>(regs, sData);
		for(int i=0; i<ELTS; i++) {
			data[blockOffset + sec + warpOffset + W*i + tid] = sData[tid*W + (tid+i)%W];
		}

		// Wait for other warps to catch back up
		for(int i=(THREADS/W)-1; i>warpId; i--) {
			__syncthreads();
		}
	}

}

#endif