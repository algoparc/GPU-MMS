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
//#include<SDL.h>
#include<iostream>
#include<fstream>
#include<vector>
#include<cmath>
#include<random>
#include<algorithm>

// Constants for window dimensions and bar graph parameters
const int SCREEN_WIDTH = 800;
const int SCREEN_HEIGHT = 600;
const int BAR_WIDTH = 20;
const int MAX_BAR_HEIGHT = 400;
const int X_OFFSET = 50;
const int Y_OFFSET = 50;

// Min function based on the given comparison function
template<typename T, fptr_t f>
__forceinline__ __device__ T myMin(T a, T b) {
  if(f(a,b)) return a;
  return b;
}

// Max function based on the given comparison function
template<typename T, fptr_t f>
__forceinline__ __device__ T myMax(T a, T b) {
  if(f(b,a)) return a;
  return b;
//  return max(a,b);
}


template<typename T, fptr_t f>
__forceinline__ __device__ void cmpSwap(T* a, T* b) {
  T temp;
  if(f(*b,*a)) {
    temp = *a;
    *a = *b;
    *b = temp;
  }
}

template<typename T, fptr_t f>
__forceinline__ __device__ void cmpSwap(T* a, T* b, bool dir) {
  T temp;
  if(f(*b,*a) ^ dir) {
    temp = *a;
    *a = *b;
    *b = temp;
  }
}

template<typename T, fptr_t f>
__forceinline__ __device__ void swap1(T* regs) {
#pragma unroll
  for(int i=0; i<31; i+=2) {
    cmpSwap<T,f>(&regs[i], &regs[i+1]);
  }
}

template<typename T, fptr_t f>
__forceinline__ __device__ void swap2(T* regs) {
  for(int i=0; i<2; i++) {
    cmpSwap<T,f>(&regs[i], &regs[2+i]);
  }
  for(int i=4; i<6; i++) {
    cmpSwap<T,f>(&regs[i], &regs[2+i]);
  }
  for(int i=8; i<10; i++) {
    cmpSwap<T,f>(&regs[i], &regs[2+i]);
  }
  for(int i=12; i<14; i++) {
    cmpSwap<T,f>(&regs[i], &regs[2+i]);
  }
  for(int i=16; i<18; i++) {
    cmpSwap<T,f>(&regs[i], &regs[2+i]);
  }
  for(int i=20; i<22; i++) {
    cmpSwap<T,f>(&regs[i], &regs[2+i]);
  }
  for(int i=24; i<26; i++) {
    cmpSwap<T,f>(&regs[i], &regs[2+i]);
  }
  for(int i=28; i<30; i++) {
    cmpSwap<T,f>(&regs[i], &regs[2+i]);
  }
}

template<typename T, fptr_t f>
__forceinline__ __device__ void swap4(T* regs) {
  for(int i=0; i<4; i++) {
    cmpSwap<T,f>(&regs[i], &regs[4+i]);
  }
  for(int i=8; i<12; i++) {
    cmpSwap<T,f>(&regs[i], &regs[4+i]);
  }
  for(int i=16; i<20; i++) {
    cmpSwap<T,f>(&regs[i], &regs[4+i]);
  }
  for(int i=24; i<28; i++) {
    cmpSwap<T,f>(&regs[i], &regs[4+i]);
  }
}

template<typename T, fptr_t f>
__forceinline__ __device__ void swap8(T* regs) {
#pragma unroll
  for(int i=0; i<8; i++) {
    cmpSwap<T,f>(&regs[i], &regs[8+i]);
  }
#pragma unroll
  for(int i=16; i<24; i++) {
    cmpSwap<T,f>(&regs[i], &regs[8+i]);
  }
}


template<typename T, fptr_t f>
__forceinline__ __device__ void swap16(T* regs) {
#pragma unroll
  for(int i=0; i<16; i++) {
    cmpSwap<T,f>(&regs[i], &regs[16+i]);
  }
}

template<typename T, fptr_t f>
__forceinline__ __device__ void swapReverse1(T* regs) {
  for(int i=0; i<29; i+=4) {
    cmpSwap<T,f>(&regs[i], &regs[i+1]);
    cmpSwap<T,f>(&regs[i+3], &regs[i+2]);
  }
}

template<typename T, fptr_t f>
__forceinline__ __device__ void swapReverse2(T* regs) {
  for(int i=0; i<2; i++) {
    cmpSwap<T,f>(&regs[i], &regs[3-i]);
  }
  for(int i=4; i<6; i++) {
    cmpSwap<T,f>(&regs[i], &regs[11-i]);
  }
  for(int i=8; i<10; i++) {
    cmpSwap<T,f>(&regs[i], &regs[19-i]);
  }
  for(int i=12; i<14; i++) {
    cmpSwap<T,f>(&regs[i], &regs[27-i]);
  }
  for(int i=16; i<18; i++) {
    cmpSwap<T,f>(&regs[i], &regs[35-i]);
  }
  for(int i=20; i<22; i++) {
    cmpSwap<T,f>(&regs[i], &regs[43-i]);
  }
  for(int i=24; i<26; i++) {
    cmpSwap<T,f>(&regs[i], &regs[51-i]);
  }
  for(int i=28; i<30; i++) {
    cmpSwap<T,f>(&regs[i], &regs[59-i]);
  }
}

template<typename T, fptr_t f>
__forceinline__ __device__ void swapReverse4(T* regs) {
  for(int i=0; i<4; i++) {
    cmpSwap<T,f>(&regs[i], &regs[7-i]);
  }
  for(int i=8; i<12; i++) {
    cmpSwap<T,f>(&regs[i], &regs[23-i]);
  }
  for(int i=16; i<20; i++) {
    cmpSwap<T,f>(&regs[i], &regs[39-i]);
  }
  for(int i=24; i<28; i++) {
    cmpSwap<T,f>(&regs[i], &regs[55-i]);
  }
}

template<typename T, fptr_t f>
__forceinline__ __device__ void swapReverse8(T* regs) {
  for(int i=0; i<8; i++) {
    cmpSwap<T,f>(&regs[i], &regs[15-i]);
  }
  for(int i=16; i<24; i++) {
    cmpSwap<T,f>(&regs[i], &regs[47-i]);
  }
}

template<typename T, fptr_t f>
__forceinline__ __device__ void swapReverse16(T* regs) {

  for(int i=0; i<16; i++) {
    cmpSwap<T,f>(&regs[i], &regs[31-i]);
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
/*
// Function to draw the bar graph
void drawBarGraph(SDL_Renderer* renderer, const std::vector<int>& data) {
    SDL_SetRenderDrawColor(renderer, 0xFF, 0xFF, 0xFF, 0xFF); // Set color to white
    SDL_RenderClear(renderer); // Clear the screen

    int x = X_OFFSET;
    for (size_t i = 0; i < data.size(); ++i) {
        int barHeight = (data[i] * MAX_BAR_HEIGHT) / std::numeric_limits<int>::max();
        int y = SCREEN_HEIGHT - barHeight - Y_OFFSET;

        SDL_Rect barRect = { x, y, BAR_WIDTH, barHeight };
        SDL_SetRenderDrawColor(renderer, 0xFF, 0x00, 0x00, 0xFF); // Set color to red
        SDL_RenderFillRect(renderer, &barRect);

        x += BAR_WIDTH + 10; // Increase x position for the next bar
    }

    SDL_RenderPresent(renderer); // Update the screen
}
*/