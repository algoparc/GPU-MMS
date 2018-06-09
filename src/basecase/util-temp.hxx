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

/* This file contains general utility functions */


#include<stdio.h>
#include<iostream>
#include<fstream>
#include<vector>
#include<cmath>
#include<random>
#include<algorithm>

template<typename T>
__forceinline__ __device__ void cmpSwap(T* a, T* b, int (*f)(T,T)) {
  T temp;
  if(f(*b,*a)) {
    temp = *a;
    *a = *b;
    *b = temp;
  }
}

template<typename T>
__forceinline__ __device__ void cmpSwap(T* a, T* b, bool dir, int (*f)(T,T)) {
  T temp;
  if(f(*b,*a) ^ dir) {
    temp = *a;
    *a = *b;
    *b = temp;
  }
}

template<typename T>
__forceinline__ __device__ void swap1(T* regs, int (*f)(T,T)) {
#pragma unroll
  for(int i=0; i<31; i+=2) {
    cmpSwap<T>(&regs[i], &regs[i+1], f);
  }
}

template<typename T>
__forceinline__ __device__ void swap2(T* regs, int (*f)(T,T)) {
  for(int i=0; i<2; i++) {
    cmpSwap<T>(&regs[i], &regs[2+i],f);
  }
  for(int i=4; i<6; i++) {
    cmpSwap<T>(&regs[i], &regs[2+i],f);
  }
  for(int i=8; i<10; i++) {
    cmpSwap<T>(&regs[i], &regs[2+i],f);
  }
  for(int i=12; i<14; i++) {
    cmpSwap<T>(&regs[i], &regs[2+i],f);
  }
  for(int i=16; i<18; i++) {
    cmpSwap<T>(&regs[i], &regs[2+i],f);
  }
  for(int i=20; i<22; i++) {
    cmpSwap<T>(&regs[i], &regs[2+i],f);
  }
  for(int i=24; i<26; i++) {
    cmpSwap<T>(&regs[i], &regs[2+i],f);
  }
  for(int i=28; i<30; i++) {
    cmpSwap<T>(&regs[i], &regs[2+i],f);
  }
}

template<typename T>
__forceinline__ __device__ void swap4(T* regs, int (*f)(T,T)) {
  for(int i=0; i<4; i++) {
    cmpSwap<T>(&regs[i], &regs[4+i],f);
  }
  for(int i=8; i<12; i++) {
    cmpSwap<T>(&regs[i], &regs[4+i],f);
  }
  for(int i=16; i<20; i++) {
    cmpSwap<T>(&regs[i], &regs[4+i],f);
  }
  for(int i=24; i<28; i++) {
    cmpSwap<T>(&regs[i], &regs[4+i],f);
  }
}

template<typename T>
__forceinline__ __device__ void swap8(T* regs, int (*f)(T,T)) {
#pragma unroll
  for(int i=0; i<8; i++) {
    cmpSwap<T>(&regs[i], &regs[8+i],f);
  }
#pragma unroll
  for(int i=16; i<24; i++) {
    cmpSwap<T>(&regs[i], &regs[8+i],f);
  }
}


template<typename T>
__forceinline__ __device__ void swap16(T* regs, int (*f)(T,T)) {
#pragma unroll
  for(int i=0; i<16; i++) {
    cmpSwap<T>(&regs[i], &regs[16+i],f);
  }
}

template<typename T>
__forceinline__ __device__ void swapReverse1(T* regs, int (*f)(T,T)) {
  for(int i=0; i<29; i+=4) {
    cmpSwap<T>(&regs[i], &regs[i+1], f);
    cmpSwap<T>(&regs[i+3], &regs[i+2], f);
  }
}

template<typename T>
__forceinline__ __device__ void swapReverse2(T* regs, int (*f)(T,T)) {
  for(int i=0; i<2; i++) {
    cmpSwap<T>(&regs[i], &regs[3-i],f);
  }
  for(int i=4; i<6; i++) {
    cmpSwap<T>(&regs[i], &regs[11-i],f);
  }
  for(int i=8; i<10; i++) {
    cmpSwap<T>(&regs[i], &regs[19-i],f);
  }
  for(int i=12; i<14; i++) {
    cmpSwap<T>(&regs[i], &regs[27-i],f);
  }
  for(int i=16; i<18; i++) {
    cmpSwap<T>(&regs[i], &regs[35-i],f);
  }
  for(int i=20; i<22; i++) {
    cmpSwap<T>(&regs[i], &regs[43-i],f);
  }
  for(int i=24; i<26; i++) {
    cmpSwap<T>(&regs[i], &regs[51-i],f);
  }
  for(int i=28; i<30; i++) {
    cmpSwap<T>(&regs[i], &regs[59-i],f);
  }
}

template<typename T>
__forceinline__ __device__ void swapReverse4(T* regs, int (*f)(T,T)) {
  for(int i=0; i<4; i++) {
    cmpSwap<T>(&regs[i], &regs[7-i],f);
  }
  for(int i=8; i<12; i++) {
    cmpSwap<T>(&regs[i], &regs[23-i],f);
  }
  for(int i=16; i<20; i++) {
    cmpSwap<T>(&regs[i], &regs[39-i],f);
  }
  for(int i=24; i<28; i++) {
    cmpSwap<T>(&regs[i], &regs[55-i],f);
  }
}

template<typename T>
__forceinline__ __device__ void swapReverse8(T* regs, int (*f)(T,T)) {
  for(int i=0; i<8; i++) {
    cmpSwap<T>(&regs[i], &regs[15-i],f);
  }
  for(int i=16; i<24; i++) {
    cmpSwap<T>(&regs[i], &regs[47-i],f);
  }
}

template<typename T>
__forceinline__ __device__ void swapReverse16(T* regs, int (*f)(T,T)) {

  for(int i=0; i<16; i++) {
    cmpSwap<T>(&regs[i], &regs[31-i],f);
  }
}


template<typename T>
__forceinline__ __device__ void reverse16Elts(T* regs) {
  T temp;
  for(int i=0; i<8; i++) {
    temp=regs[i];
    regs[i] = regs[15-i];
    regs[15-i] = temp;
  }
}

template<typename T>
__forceinline__ __device__ void reverse32Elts(T* regs) {
  T temp;
  for(int i=0; i<16; i++) {
    temp=regs[i];
    regs[i] = regs[31-i];
    regs[31-i] = temp;
  }
}
