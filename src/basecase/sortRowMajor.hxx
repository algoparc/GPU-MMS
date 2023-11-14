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

/* This file contains the functions for the base-case sorting of 1024 elements,
as well as the functions used to merge pairs of nodes of the minBlockHeap structure */


#include<stdio.h>
#include<iostream>
#include<fstream>
#include<vector>
#include<cmath>
#include<random>
#include<algorithm>
#include "../params.h"
#include "util.hxx"
//#include "basesort.h"
//#include "../cmp.hxx"

void test_shflMerge();

// Wrapper around shfl command for key-value pairs
template<typename T>
__forceinline__ __device__ T shfl_wrapper(T val, int dist, int width) {
  T tempObj;
  tempObj = SHFL_XOR((CASTTYPE)val, dist, width);
  return tempObj;
}

template<typename T, fptr_t f>
__forceinline__ __device__ void merge32dir(T* regs, bool order, bool alt);
template<typename T, fptr_t f>
__forceinline__ __device__ void merge64dir(T* regs, bool order, bool alt);
template<typename T, fptr_t f>
__forceinline__ __device__ void merge128dir(T* regs, bool order, bool alt);
template<typename T, fptr_t f>
__forceinline__ __device__ void merge256dir(T* regs, bool order, bool alt);

template<typename T, fptr_t f>
__forceinline__ __device__ void merge32(T* regs, bool alt);
template<typename T, fptr_t f>
__forceinline__ __device__ void merge64(T* regs, bool alt);
template<typename T, fptr_t f>
__forceinline__ __device__ void merge128(T* regs, bool alt);
template<typename T, fptr_t f>
__forceinline__ __device__ void merge256(T* regs, bool alt);
template<typename T, fptr_t f>
__forceinline__ __device__ void merge512(T* regs, bool alt);

template<typename T, fptr_t f>
__forceinline__ __device__ void bitonicMerge32(T* regs, bool dir);

template<typename T, fptr_t f>
__forceinline__ __device__ void bitonicSort32(T* regs, bool dir);


// Sort 1024 elements: Each thread sorts a row, then performs bitonic merge using shfl
template<typename T, fptr_t f>
__forceinline__ __device__ void sortSquareRowMajor(T* regs, int dir) {
    bitonicSort32<T,f>(regs, threadIdx.x%2); // Each thread sorts row in alternating order

// Bitonic merge network
    merge32<T,f>(regs,dir);
    merge64<T,f>(regs,dir);
    merge128<T,f>(regs,dir);
    merge256<T,f>(regs,dir);
    merge512<T,f>(regs,dir);
}

template<typename T, fptr_t f>
__forceinline__ __device__ void bitonicMerge32(T* regs, bool dir) {

  swap16<T,f>(regs);
  swap8<T,f>(regs);
  swap4<T,f>(regs);
  swap2<T,f>(regs);
  swap1<T,f>(regs);

  if(dir) reverse32Elts<T>(regs);
}

template<typename T, fptr_t f>
__forceinline__ __device__ void bitonicSort32(T* regs, bool dir) {
  swap1<T,f>(regs);

  swapReverse2<T,f>(regs);
  swap1<T,f>(regs);
 
  swapReverse4<T,f>(regs);
  swap2<T,f>(regs);
  swap1<T,f>(regs);

  swapReverse8<T,f>(regs);
  swap4<T,f>(regs);
  swap2<T,f>(regs);
  swap1<T,f>(regs);

  swapReverse16<T,f>(regs);
  swap8<T,f>(regs);
  swap4<T,f>(regs);
  swap2<T,f>(regs);
  swap1<T,f>(regs);

  if(dir) reverse32Elts<T>(regs);
}

// mergeX means merging two arrays of X to form one large array of 2X items
// down evaluates to true if threadID is is even, otherwise false
// order evaluates to true if the pair of threads 2 bit is on. i.e., if the thread IDs are 2,3,6,7,10,11,14,15,18,19,22,23,26,27,30,31
// down ^ order evaluates to true if exactly one of down or order are true
// In this case, down ^ order is true if thread ID is == 3 (mod 4) or 0 (mod 4)
// if alt is true, then down evaluates to true if threadID is odd
// then down ^ order is true if thread ID is 1 (mod 4) or 2 (mod 4)

// For the successive mergeX, for X in 64,128,256,512, every 2,4,8,16 threads, respectively, have the same behavior
// In other words, at the end of each merge32 operation, every pair of threads will have the same values, and will have the same behavior for merge64
// At the end of the merge64 operation, every 4 threads will have the same values, and will have the same behavior for merge128
// and so on and so forth. By the end of the merge512 operation, all threads will have the exact same values in their registers
template<typename T, fptr_t f>
__forceinline__ __device__ void merge32(T* regs, bool alt) {
  bool down = (threadIdx.x < (threadIdx.x ^ 1))^alt;
  bool order = (threadIdx.x > (threadIdx.x ^ 2));
  T temp;
  T val;

  for(int i=0; i<ELTS; i++) {
    val=regs[i];
    temp = shfl_wrapper(val, 1, W); // get the item from the register a distance 1 away. Essentially, compare (0, 1), (2, 3), (4, 5), ...
    if(down ^ order)
      regs[i] = myMin<T,f>(val,temp);
    else
      regs[i] = myMax<T,f>(val,temp);
  }
  bitonicMerge32<T,f>(regs, order);
}

template<typename T, fptr_t f>
__forceinline__ __device__ void merge64(T* regs, bool alt) {
  bool down = (threadIdx.x < (threadIdx.x ^ 2))^alt;
  bool order = (threadIdx.x > (threadIdx.x ^ 4));
  T temp;
  T val;

  for(int i=0; i<ELTS; i++) {
    val=regs[i];
    temp = shfl_wrapper(val, 2, W);
    if(down ^ order)
      regs[i] = myMin<T,f>(val, temp);
    else
      regs[i] = myMax<T,f>(val,temp);
  }
  merge32dir<T,f>(regs, order, alt); 
}

template<typename T, fptr_t f>
__forceinline__ __device__ void merge128(T* regs, bool alt) {
  bool down = (threadIdx.x < (threadIdx.x ^ 4))^alt;
  bool order = (threadIdx.x > (threadIdx.x ^ 8));
  T temp;
  T val;

  for(int i=0; i<ELTS; i++) {
    val=regs[i];
    temp = shfl_wrapper(val, 4, W);
    if(down ^ order)
      regs[i] = myMin<T,f>(val, temp);
    else
      regs[i] = myMax<T,f>(val,temp);
  }
  merge64dir<T,f>(regs, order, alt); 
}

template<typename T, fptr_t f>
__forceinline__ __device__ void merge256(T* regs, bool alt) {
  bool down = (threadIdx.x < (threadIdx.x ^ 8))^alt;
  bool order = (threadIdx.x > (threadIdx.x ^ 16));
  T temp;
  T val;

  for(int i=0; i<ELTS; i++) {
    val=regs[i];
    temp = shfl_wrapper(val, 8, W);
    if(down ^ order)
      regs[i] = myMin<T,f>(val, temp);
    else
      regs[i] = myMax<T,f>(val,temp);
  }
  merge128dir<T,f>(regs, order, alt); 
}

template<typename T, fptr_t f>
__forceinline__ __device__ void merge512(T* regs, bool dir) {
  bool down = (threadIdx.x < (threadIdx.x ^ 16)) ^ dir;
  T temp;
  T val;

  for(int i=0; i<ELTS; i++) {
    val=regs[i];
    temp = shfl_wrapper(val, 16, W);
    if(down)
      regs[i] = myMin<T,f>(val, temp);
    else
      regs[i] = myMax<T,f>(val,temp);
  }
  merge256dir<T,f>(regs, false, dir); 
}


template<typename T, fptr_t f>
__forceinline__ __device__ void merge32dir(T* regs, bool order, bool alt) {
  T temp,val;
  bool down = (threadIdx.x < (threadIdx.x ^ 1))^alt;

  for(int i=0; i<ELTS; i++) {
    val=regs[i];
    temp = shfl_wrapper(val, 1, W);
    if(down ^ order)
      regs[i] = myMin<T,f>(val, temp);
    else
      regs[i] = myMax<T,f>(val,temp);
//    down = down^alt;
  }
  bitonicMerge32<T,f>(regs, order^alt);
}

template<typename T, fptr_t f>
__forceinline__ __device__ void merge64dir(T* regs, bool order, bool alt) {
  T temp,val;
  bool down = (threadIdx.x < (threadIdx.x ^ 2))^alt;

  for(int i=0; i<ELTS; i++) {
    val=regs[i];
    temp = shfl_wrapper(val, 2, W);
    if(down ^ order)
      regs[i] = myMin<T,f>(val, temp);
    else
      regs[i] = myMax<T,f>(val,temp);

//    down = down^alt;
  }
//  bubbleThreadSort(regs, order);
  merge32dir<T,f>(regs, order, alt);
}

template<typename T, fptr_t f>
__forceinline__ __device__ void merge128dir(T* regs, bool order, bool alt) {
  T temp,val;
  bool down = (threadIdx.x < (threadIdx.x ^ 4)) ^ alt;

  for(int i=0; i<ELTS; i++) {
    val=regs[i];
    temp = shfl_wrapper(val, 4, W);
    if(down ^ order)
      regs[i] = myMin<T,f>(val, temp);
    else
      regs[i] = myMax<T,f>(val,temp);
//    down = down^alt;
  }
//  bubbleThreadSort(regs, order);
  merge64dir<T,f>(regs, order, alt);
}

template<typename T, fptr_t f>
__forceinline__ __device__ void merge256dir(T* regs, bool order, bool alt) {
  T temp,val;
  bool down = (threadIdx.x < (threadIdx.x ^ 8))^alt;

  for(int i=0; i<ELTS; i++) {
    val=regs[i];
    temp = shfl_wrapper(val, 8, W);
    if(down)
      regs[i] = myMin<T,f>(val, temp);
    else
      regs[i] = myMax<T,f>(val,temp);
//    down = down^alt;
  }
//  bubbleThreadSort(regs, order);
  merge128dir<T,f>(regs, order, alt);
}

template<typename T, fptr_t f>
__forceinline__ __device__ void merge512dir(T* regs, bool order, bool alt) {
  bool down = (threadIdx.x < (threadIdx.x ^ 16));
  T temp;
  T val;

  for(int i=0; i<ELTS; i++) {
    val=regs[i];
    temp = shfl_wrapper(val, 16, W);
    if(down ^ order)
      regs[i] = myMin<T,f>(val, temp);
    else
      regs[i] = myMax<T,f>(val,temp);
    down = down^alt;
  }
  merge256dir<T,f>(regs, order, alt); 
}

template<typename T, fptr_t f>
__forceinline__ __device__ void shflSortCols(T* regs) {
  merge32<T,f>(regs,false);
  merge64<T,f>(regs,false);
  merge128<T,f>(regs,false);
  merge256<T,f>(regs,false);
}

template<typename T, fptr_t f>
__forceinline__ __device__ void shflSortCols32(T* regs) {
  merge32<T,f>(regs,false);
  merge64<T,f>(regs,false);
  merge128<T,f>(regs,false);
  merge256<T,f>(regs,false);
  merge512<T,f>(regs,false);
}

// Each thread writes 32 elements in diagonal to avoid bank conflicts and accomplish a transpose
template<typename T>
__device__ void transposeSquares(T* regs, T* sData) {
  int tid = threadIdx.x &(W-1);
#pragma unroll
  for(int i=0; i<ELTS; i++) {
    sData[((tid+i)&(W-1)) + (W)*i] = regs[i]; 
  }
}

// Bitonic merge network between 2 warps, each having 1024 elements
template<typename T, fptr_t f>
__forceinline__ __device__ void warpSwap2(T* regs, T* sData, bool dir) {
  int tid=threadIdx.x&(W-1);
  int warpId = threadIdx.x/W;
  for(int i=0; i<ELTS; i++) {
    sData[i*W+warpId*1024+tid] = regs[i];
  }
  bool down = (warpId < (warpId^1));
  int neighborId = warpId+down-(!down);

  __syncthreads();
  for(int i=0; i<ELTS; i++) {
    if(down ^ dir) {
      regs[i] = myMin<T,f>(regs[i], sData[i*W+neighborId*1024+tid]);
    }
    else {
      regs[i] = myMax<T,f>(regs[i], sData[i*W+neighborId*1024+tid]);
    }
  }
  merge512<T,f>(regs,dir); // Continue with the bitonic network within warp
}

// Bitonic merge network between 4 warps, each having 1024 elements
template<typename T, fptr_t f>
__forceinline__ __device__ void warpSwap4(T* regs, T* sData) {
  int SQUARES=4;
  int tid=threadIdx.x&(W-1);
  int warpId = threadIdx.x/W;
  for(int i=0; i<ELTS; i++) {
    sData[i*W+warpId*1024+tid] = regs[i];
  }
  bool down2 = (warpId < (warpId ^ 3));
  bool down = (warpId < (warpId ^ 1));
  int temp;
  __syncthreads();
  for(int i=0; i<ELTS; i++) {
    if(down2) {
      regs[i] = myMin<T,f>(regs[i], sData[i*W+(warpId+2)*1024+tid]);
      temp = myMin<T,f>(sData[i*W+(warpId+1)*1024+tid], sData[i*W+((warpId+3)&(SQUARES-1))*1024+tid]);
    }
    else {
      regs[i] = myMax<T,f>(regs[i], sData[i*W+((warpId+2)&(SQUARES-1))*1024+tid]);
      temp = myMax<T,f>(sData[i*W+(warpId-1)*1024+tid], sData[i*W+((warpId+1)&(SQUARES-1))*1024+tid]);
    }
    if(down) {
      regs[i] = myMin<T,f>(regs[i], temp);
    }
    else
      regs[i] = myMax<T,f>(regs[i],temp);
  }
  merge512<T,f>(regs,false); // Continue with bitonic network within warp
}


