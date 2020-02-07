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

/* Main basecase sorting Kernel */
template<typename T, fptr_t f>
__global__ void squareSort(T* data, int N) {
	T regs[ELTS];


	int blockOffset = (N/gridDim.x)*blockIdx.x;
	// printf("%d\t%d\n", blockOffset, blockIdx.x);


	// N/gridDim.x always comes out to 1024?, sec always increments by M
	for(int sec = 0; sec < (N/gridDim.x); sec += M*(THREADS/W)) {
		// printf("%d\n", threadIdx.x);
		// i goes from 0 to 31 and increments by 1
		for(int i=0; i<ELTS; i++) {
			// printf("%d\n", blockOffset + sec + (i*THREADS) + threadIdx.x);
			if ((blockOffset + sec + (i*THREADS) + threadIdx.x) >= N) {
				// printf("dataIndex\t%d\tdata\t%d\n", (blockOffset + sec + (i*THREADS) + threadIdx.x), data[blockOffset + sec + (i*THREADS) + threadIdx.x]);
				regs[i] = 2147483647;
			}
			else {
				// printf("%d\n", data[blockOffset + sec + (i*THREADS) + threadIdx.x]);
				regs[i] = data[blockOffset + sec + (i*THREADS) + threadIdx.x];
			}
			
			// for (int j = 0; j < ELTS; j++) {
			// 	printf("regs[%d]\t%d\n", j, regs[j]);
			// }
			
		}
		
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

		sortSquareRowMajor<T,f>(regs, false);
		// printf("BREAK\n");
		// for (int i = 0; i < ELTS; i++) {
		// 	printf("%d\t%d\t%d\n", blockOffset, tid, regs[i]);
		// }
		// printf("\n");

// Warps within a block use the same shared memory, so they have to take turns transposing
// This lets us have more warps and increases performance!
		for(int i=0; i<warpId; i++)
			__syncthreads();

// transpose in shared memory then write to global
		transposeSquares<T>(regs, sData);
		for(int i=0; i<ELTS; i++) {
			// printf("%d:\tsData: %d\n", i, sData[tid*W + (tid+i)%W]);
			data[blockOffset + sec + warpOffset + W*i + tid] = sData[tid*W + (tid+i)%W];
			// printf("%d:\tdata: %d\n", i, data[blockOffset + sec + warpOffset + W*i + tid]);
		}

		for(int i=(THREADS/W)-1; i>warpId; i--) // Wait for other warps to catch back up
			__syncthreads();
	}

}

#endif