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

#ifndef testMM_cu
	#define testMM_cu
	#include <stdio.h>
	#include <iostream>
	#include <fstream>
	#include <vector>
	#include <cmath>
	#include <random>
	#include <algorithm>
	#include "multimergesort.hxx"
	#include "buildData.h"
#endif


/* CPU FUNCTION HEADERS*/
template<typename T>
void test_multimergesort(int p, int N);


int main(int argc, char** argv) {

	if(argc != 2) {
	printf("Usage: ./testMM <N>\n");
	exit(1);
	}

	int N = atoi(argv[1]);
	test_multimergesort<DATATYPE>(BLOCKS, N);

	return 0;
}

// Create random data and sort it...
template<typename T>
void test_multimergesort(int p, int N) {

	// Initialize variables used to measure the runtime of the program.
	cudaEvent_t start, stop;
	float time_elapsed=0.0;
	float minTime=99999;
	float maxTime=0.0;
	float total_time=0.0;


	/*	Possible fix: Figure out if the list needs to be padded with extra values
	 *	so that its size will match M * K^i (only necessary for N > 1024 (M?)).
	 *	If it needs to be padded, keep the next highest value of M * K^i.
	 */
	int new_N = N; // make a new value of N for padding
	#ifdef USE_PADDING
	if (N > M) {
		// Keep incrementing new_N by a factor of K until it is greater or equal to N
		new_N = M * K;
		while (new_N < N) {
			new_N *= K;
		}
	}
	#endif

	#ifdef PRINT_DEBUG
	// print the adjusted (new) value of N for comparison
	printf("N: %d\tnew_N: %d\tPadded Elements: %d\n", N, new_N, new_N - N);
	#endif

	// Create sample sorted lists
	T* h_data = (T*)malloc(new_N*sizeof(T));

	// Allocate space for input and output lists on the GPU
	T* d_data;
	T* d_output;
	cudaMalloc(&d_data, new_N*sizeof(T));
	cudaMalloc(&d_output, new_N*sizeof(T));

	// srand(time(NULL)); // pseudo-random seeding using time
	srand(11); // consistent seeding at 11 for testing
	for(int it=0; it<ITERS; it++) {

		// Create random list to be sorted
		create_random_list<T>(h_data, N, 0);
		#ifdef USE_PADDING
		// Pad the list with (new_N - N) elements
		pad_list<T>(h_data, N, new_N);
		#endif

		#ifdef PRINT_DEBUG
		// Print the unsorted, padded list
		for (int i = 0; i < new_N; i++) {
			printf("%d\t%d\n", i, h_data[i]);
		}
		#endif


		// Copy list to GPU
		cudaMemcpy(d_data, h_data, new_N*sizeof(T), cudaMemcpyHostToDevice);

		// Zero out result array
		cudaMalloc(&d_output, new_N*sizeof(T));
		//  cudaMemset(&d_output, 0, N*sizeof(T));

		cudaDeviceSynchronize();
		// Timer functions
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);

		// Run GPU-MMS.  T is datatype and cmp is comparison function (defined in cmp.hxx)
		d_output = multimergesort<T,cmp>(d_data, d_output, h_data, p, new_N);
		cudaDeviceSynchronize();

		cudaEventRecord(stop,0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time_elapsed, start, stop);
		total_time += time_elapsed;
		if(time_elapsed < minTime) minTime = time_elapsed;
		if(time_elapsed > maxTime) maxTime = time_elapsed;
		time_elapsed=0.0;

		if(it<ITERS-1)
			cudaFree(&d_output);
	}
	// Calculate the average time to sort a list.
	float average_time = total_time/ITERS;
	printf("Sorted %d input(s) of size %d\nTotal Time: %lf\nAverage Time: %lf\nMin Time: %lf\nMax Time: %lf\n", ITERS, N, total_time, average_time, minTime, maxTime);

	// copy sorted result back to CPU
	cudaMemcpy(h_data, d_output, N*sizeof(T), cudaMemcpyDeviceToHost);

	// If debug mode is on, check that output is correct
#ifdef DEBUG
	/*	Loop through the list and look for any adjacent elements that
	 *	are not sorted in increasing order. Print these errors.
	 *	Note: this only checks the last list if there are multiple (ITERS > 1).
	 */
	bool error=false;
	for(int i=1; i<N; i++) {
		if(host_cmp<int>(h_data[i], h_data[i-1])) {
			printf("ERROR FOUND!\n");
			printf("h_data[%d]: %d\n", i, h_data[i]);
			printf("h_data[%d]: %d\n", i-1, h_data[i-1]);
			error=true;
		}
	}

	if(error) {
		printf("NOT SORTED!\n");
	}
	else {
		#ifdef PRINT_DEBUG
		// If the array is sorted, print it out.
		for (int i = 0; i < N; i++) {
			printf("%d\t%d\n", i, h_data[i]);
		}
		#endif
		printf("SORTED!\n");
	}
#endif

	// Free dynamically allocated memory
	cudaFree(d_data);
	cudaFree(d_output);
	free(h_data);
}


// Function to test just the basecase method
template<typename T>
void test_squareSort(int N) {

	cudaEvent_t start, stop;
	float time_elapsed=0.0;
	float total_time=0.0;

	T* h_data = (T*)malloc(N*sizeof(T));
	T* d_data;
	cudaMalloc(&d_data, N*sizeof(T));
	srand(time(NULL));

	for(int it=0; it < ITERS; it++) {
		create_random_list<T>(h_data, N, 0);

		cudaMemcpy(d_data, h_data, N*sizeof(T), cudaMemcpyHostToDevice);

		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);

		squareSort<T><<<BLOCKS,THREADS>>>(d_data);
		cudaDeviceSynchronize();
		cudaEventRecord(stop,0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time_elapsed, start, stop);

		total_time += time_elapsed;
	}

	printf("%lf ", total_time/ITERS);
	printf("\n");

	cudaMemcpy(h_data, d_data, N*sizeof(T), cudaMemcpyDeviceToHost);

	bool sorted=true;
	for(int j=0; j<N; j+=M) {
		for(int i=1; i<M; i++) {
			if(cmp(h_data[i+j], h_data[j+i-1]))
				sorted=false;
		} 
	}
	if(!sorted) printf("NOT SORTED\n");
}
