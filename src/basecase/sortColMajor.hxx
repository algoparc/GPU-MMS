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


/* This file contains depricated code that is not used.  It was tried
as an alternative implementation but under-performed and thus is not used. */

// Perform bitonic merge network on each column using shfl
__forceinline__ __device__ void bitonicMergeCol(int* regs, int alt) {
  bool down;
  int temp;
  int val;
  for(int dist=ELTS/2; dist > 0; dist/=2) {
    down = (threadIdx.x < (threadIdx.x ^ dist));
    for(int i=0; i<ELTS; i++) {
      val=regs[i];
      temp = __shfl_xor(val, dist, W);
      if(down)
        regs[i] = min(val, temp);
      else
        regs[i] = max(val,temp);
      if(i%alt == alt-1)
        down = !down;
    }
  }
}

// Perform bitonic sorting network on each column using shfl
__forceinline__ __device__ void bitonicSortCol(int* regs, int alt) {
  bool down, dir;
  int temp;
  int val;
 
  for(int step=2; step<=ELTS; step*=2) {
    dir = (threadIdx.x < (threadIdx.x ^ step));
    for(int dist=step/2; dist > 0; dist/=2) {
      down = (threadIdx.x < (threadIdx.x ^ dist));
      for(int i=0; i<ELTS; i++) {
        val=regs[i];
        temp = __shfl_xor(val, dist, W);
        if(down^dir)
          regs[i] = min(val, temp);
        else
          regs[i] = max(val,temp);
        down = down^alt;
      }
    }
  }
}


/* Tried using this to avoid having to traspose, but was much slower so don't use it*/
__forceinline__ __device__ void sortSquareColMajor(int* regs) {
    bitonicSortRow(regs, true);
    swapReverse1(regs);
    bitonicMergeCols(regs, 2);
    swap2(regs);
    swap1(regs);
    bitonicMergeCols(regs, 4);
    swapReverse4(regs);
    swap2(regs);
    swap1(regs);
    bitonicMergeCols(regs,8);
    swapReverse8(regs);
    swap4(regs);
    swap2(regs);
    swap1(regs);
    bitonicMergeCols(regs,16);
    swapReverse16(regs);
    swap8(regs);
    swap4(regs);
    swap2(regs);
    swap1(regs);
    bitonicMergeCols(regs,32);
