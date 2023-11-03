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


#define K 4 // Branching factor (lists merged per round)
#define PL 2 // Log of K
#define PIPELINE 1 // Set to 1 to enable pipelining heap merging

#define M 1024 // Size of base case (keep at 1024)
#define THREADS 32 // Threads per block
#define W 32
#define B 32
#define LOGB 5
#define RANGE 1048576 // Range of randomly generated values
#define ELTS 32

#define FULL_MASK 0xFFFFFFFF

#if defined(__CUDACC_VER_MAJOR__) && defined(__CUDACC_VER_MINOR__)
// Check if the CUDA version is 9.0 or later
#if (__CUDACC_VER_MAJOR__ >= 9)
#define SHFL_XOR(val, delta, width) __shfl_xor_sync(FULL_MASK, val, delta, width)
#else
#define SHFL_XOR(val, delta, width) __shfl_xor(val, delta, width)
#endif
#else
// Fallback if CUDA version macros are not defined
#define SHFL_XOR(val, delta) __shfl_xor(val, delta)
#endif
