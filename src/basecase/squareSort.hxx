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
#include "../params.h"
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

  for(int sec = 0; sec < (N/gridDim.x); sec += M*(THREADS/W)) { // iterates until you hit the upper bound of N/gridDim.x which is the number of ELTS processed per block 
    for(int i=0; i<ELTS; i++) {
      regs[i] = data[blockOffset + sec + (i*THREADS) + threadIdx.x];
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

// Warps within a block use the same shared memory, so they have to take turns transposing
// This lets us have more warps and increases performance!
    for(int i=0; i<warpId; i++)
      __syncthreads();

// transpose in shared memory then write to global
    transposeSquares<T>(regs, sData);
    for(int i=0; i<ELTS; i++) {
      data[blockOffset + sec + warpOffset + W*i + tid] = sData[tid*W + (tid+i)%W];
    }

    for(int i=(THREADS/W)-1; i>warpId; i--) // Wait for other warps to catch back up
      __syncthreads();
  }
}

