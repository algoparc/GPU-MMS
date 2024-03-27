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


#define DEBUG 1 // Set shis to 1 to check that the output is correctly sorted
#define PRINT 0 // Set this to 1 to print first M elements of the array for further debugging
#define ITERS 1 // Number of iterations to compute average runtime
#define BLOCKS 128

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include "multimergesort.hxx"
#include "buildData.h"

#include <stdlib.h>

/* CPU FUNCTION HEADERS*/
template <typename T>
void test_multimergesort(int p, int N);
template <typename T>
void test_deterministic_mergelevels(int p, int N);
template <typename T>
void test_squareSort(int N);
template <typename T>
void CPUsort(T *arr, int N);
template <typename T, fptr_t f>
void test_arrayEquality(T *arr1, T *arr2, int N);
template <typename T, fptr_t f>
void selection_sort(T* a, int N);
void printArr(int* arr, int N);

int main(int argc, char **argv)
{

  if (argc != 2){
    printf("Usage: ./testMM <N>\n");
    exit(1);
  }

  int N = atoi(argv[1]);
  test_multimergesort<DATATYPE>(BLOCKS, N);
  // test_deterministic_mergelevels<DATATYPE>(BLOCKS, N);
  // test_squareSort<DATATYPE>(N);

  return 0;
}

template <typename T>
void test_deterministic_mergelevels(int p, int N)
{
  cudaError_t err;
  cudaEvent_t start, stop;
  float time_elapsed = 0.0;
  float minTime = 99999;
  float maxTime = 0.0;

  // Create sample sorted lists
  
  int padding = (N%M) ? M-N%M : 0;
  T *h_data = (T *)malloc((N+padding) * sizeof(T));
  T *h_output = (T *)malloc((N+padding) * sizeof(T));
  for (int i = 0; i < padding; i++){
    *(h_data + i) = MAXVAL - 1;
  }

  T *d_data;
  T *d_output;
  cudaMalloc(&d_data, (N+padding) * sizeof(T));
  cudaMalloc(&d_output, (N+padding) * sizeof(T));

  srand(0); // time(NULL)



  // Create random list to be sorted
  create_random_list<T>(h_data, N, 0);

  for (int it = 0; it < 2; it++) {

    // Create sorted initial list to test
    // create_sorted_list<T>(h_data, N, 0);
    
    // Copy list to GPU
    cudaMemcpy(d_data, h_data, N * sizeof(T), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    // Run GPU-MMS.  T is datatype and cmp is comparison function (defined in cmp.hxx)
    if (padding)
      pad<T><<<1,padding>>>(d_data+N, MAXVAL - 1);
    d_output = multimergesort<T, cmp>(d_data, d_output, h_data, p, N+padding);
      
    cudaDeviceSynchronize();

    if (it < ITERS - 1)
      cudaFree(&d_output);
    
    // copy sorted result back to CPU
    if (it == 0)
      cudaMemcpy(h_output, d_output, N * sizeof(T), cudaMemcpyDeviceToHost);
  }

  // copy sorted result back to CPU
  cudaMemcpy(h_data, d_output, N * sizeof(T), cudaMemcpyDeviceToHost);

  test_arrayEquality<T, host_cmp>(h_data, h_output, N);


  cudaFree(d_data);
  cudaFree(d_output);
  free(h_data);
  free(h_output);
}


// Create random data and sort it...
template <typename T>
void test_multimergesort(int p, int N)
{
  cudaEvent_t start, stop;
  float time_elapsed = 0.0;
  float minTime = 99999;
  float maxTime = 0.0;

  // Create sample sorted lists
  
  int padding = (N%M) ? M-N%M : 0;
  T *h_data = (T *)malloc((N+padding) * sizeof(T));
  for (int i = 0; i < padding; i++){
    *(h_data + i) = MAXVAL - 1;
  }

  T *d_data;
  T *d_output;
  cudaMalloc(&d_data, (N+padding) * sizeof(T));
  cudaMalloc(&d_output, (N+padding) * sizeof(T));
  float total_time = 0.0;

  srand(0); // time(NULL)

  for (int it = 0; it < ITERS; it++) {

    // Create random list to be sorted
    // create_random_list<T>(h_data, N, 0);

    // Create sorted initial list to test
    create_sorted_list<T>(h_data, N);
    // randomize<T>(h_data, N);
    
    // Copy list to GPU
    cudaMemcpy(d_data, h_data, N * sizeof(T), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    // Timer functions
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Run GPU-MMS.  T is datatype and cmp is comparison function (defined in cmp.hxx)
    if (padding)
      pad<T><<<1,padding>>>(d_data+N, MAXVAL - 1);
    d_output = multimergesort<T, cmp>(d_data, d_output, h_data, p, N+padding);
    cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_elapsed, start, stop);
    total_time += time_elapsed;
    if (time_elapsed < minTime)
      minTime = time_elapsed;
    if (time_elapsed > maxTime)
      maxTime = time_elapsed;
    time_elapsed = 0.0;

    if (it < ITERS - 1)
      cudaFree(&d_output);
  }
  total_time = total_time / ITERS;
  printf("%lf %lf %lf\n", total_time, minTime, maxTime);

  // copy sorted result back to CPU
  cudaMemcpy(h_data, d_output, N * sizeof(T), cudaMemcpyDeviceToHost);

// If debug mode is on, check that output is correct
#ifdef DEBUG
  bool error = false;
  int erroneous_index;
  for (int i = 1; i < N; i++){
    if (host_cmp<int>(h_data[i], h_data[i - 1])){
      error = true;
      erroneous_index = i;
      break;
    }
  }
  
  if (error)
    printf("NOT SORTED! Item at index %d is less than its predecessor.\n", erroneous_index);
  else
    printf("SORTED!\n");
  

#if PRINT == 1
  printf("[%d", h_data[0]);
  for (int i = 1; i < M; i++)
    printf(", %d", h_data[i]);
  printf("]\n");
#endif
#endif

  cudaFree(d_data);
  cudaFree(d_output);
  free(h_data);
}

// Function to test just the basecase method
template <typename T>
void test_squareSort(int N){
  cudaEvent_t start, stop;
  float time_elapsed = 0.0;
  float total_time = 0.0;

  T *h_data = (T *)malloc(N * sizeof(T));
  T *cpu_data = (T *)malloc(N * sizeof(T));
  T *d_data;
  cudaMalloc(&d_data, N * sizeof(T));
  srand(time(NULL));

  for (int it = 0; it < ITERS; it++)
  {
    create_random_list<T>(h_data, N);
    randomize<T>(h_data, N);

    cudaMemcpy(d_data, h_data, N * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(cpu_data, h_data, N * sizeof(T), cudaMemcpyHostToHost);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    squareSort<T, cmp><<<(((N+M-1)/M) / (THREADS/W)), THREADS>>>(d_data, N); // number of blocks was initially BLOCKS, but that was incorrect. squareSort is implemented to have grid dimensions that scale with input size

    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_elapsed, start, stop);

    total_time += time_elapsed;
  }

  printf("%lf ", total_time / ITERS);
  printf("\n");
  cudaError_t err = cudaGetLastError();
  if (err)
    printf("%s\n", cudaGetErrorString(err));

  cudaMemcpy(h_data, d_data, N * sizeof(T), cudaMemcpyDeviceToHost);
  cudaFree(d_data);
#ifdef DEBUG
  bool sorted = true;

  int start_of_base_case;
  for (start_of_base_case=0; start_of_base_case<=N-M; start_of_base_case += M)
    selection_sort<int, host_cmp>(cpu_data + start_of_base_case, M);
  selection_sort<int, host_cmp>(cpu_data + start_of_base_case, N-start_of_base_case);
  test_arrayEquality<int, host_cmp>(cpu_data, h_data, M*(N/M)); // Compare a number of elements equal to the greatest multiple of M.
#if PRINT == 1
  printf("[%d", h_data[0]);
  for (int i = 1; i < N; i++)
    printf(", %d", h_data[i]);
  printf("]\n");
#endif

  for (int j = 0; j < N; j += M){
    for (int i = 1; i < M; i++){
      if (i + j < N && host_cmp<int>(h_data[i + j], h_data[j + i - 1]))
        sorted = false;
    }
  }
  if (!sorted)
    printf("NOT SORTED\n");
  else
    printf("SORTED!\n");
    
#endif
}

template <typename T, fptr_t f>
void test_arrayEquality(T *arr1, T *arr2, int N){
  bool equal = true;
  int index = -1;
  for (int i=0; i<N; i++){
    if (arr1[i] != arr2[i]){
      equal = false;
      index = i;
      break;
    }
  }
  if (!equal)
    printf("NOT EQUAL! DIFFERING VALUES BEGINNING AT INDEX %d!\n", index);
  else
    printf("EQUAL!\n");
}

template <typename T, fptr_t f>
void selection_sort(T* a, int N){
  int temp;
  for (int i=0; i<N; i++){
    for (int j=i+1; j<N; j++){
      if (f(a[j],a[i])){
        temp = a[i];
        a[i] = a[j];
        a[j] = temp;
      }
    }
  }
}

void printArr(int* arr, int N){
  printf("[%d", *arr);
  for (int i = 1; i < N; i++){
    printf(", %d", *(arr+i));
  }
  printf("]\n");
}