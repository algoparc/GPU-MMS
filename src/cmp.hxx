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


/* This file is where you define the datatype and comparison operator to sort the given input data. */

#define DATATYPE int // Datatype to sort
#define CASTTYPE int // Datatype to cast key to in order to use __shfl()


// Define MAX_INT and MIN_INT as maximum and minimum key values
// FOR INTEGERS
#define MAXVAL 2147483647
#define MINVAL -2147483647
// FOR LONGS
//#define MAXVAL 0x0000ffff00000001
//#define MINVAL 0x000000010000ffff 

// Define abstract format of comparison function
typedef int(*fptr_t)(DATATYPE, DATATYPE);

// Comparison function used to sort by.
// Edit this to be whatever comparison function is needed.
template<typename T>
__forceinline__ __device__ int cmp(T a, T b) {
  return a < b; // Basic less than comparison

// L1-norm (manhattan distance)
//  return ((((int)a)+(int)(a>>32)) < (((int)b)+(int)(b>>32))); // L1-norm (manhattan distance)

// Comparison of fractional values without loss of precision
//  return (((int)(a>>32)*(int)b) < (((int)(b>>32))*(int)a));
}

// Host comparison function to check correctness when debugging
template<typename T>
__forceinline__ int host_cmp(T a, T b) {
  return a < b;
}


/*******************************************************************
* Example structure for key/value pair sorting
*******************************************************************/
/*
struct testType {
  volatile int key; // Keys by which to compare objects

// Pointer to the objects to be sorted (defined as index within an array).  Because GPU memory is limited, 32-bit integer index suffices to cover address space
  volatile int val; 

// Casting functions to enable use of a single __shfl operation
  __forceinline__ __device__ operator CASTTYPE() { return ((long)key + ((long)((long)val << 32)));}
  __forceinline__ __device__ testType& operator =(long &newVal) { key = (int)newVal; val = (int)((long)newVal >> 32); return *this;}
  __forceinline__ __device__ testType& operator =(const long &newVal) { key = (int)newVal; val = (int)((long)newVal >> 32); return *this;}

  __forceinline__ __device__ testType& operator =(const int &newVal) { key = newVal; val = newVal; return *this;}
};
*/

// Variable used for debugging and performance counting
__device__ int tot_cmp;
