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
	using namespace std;
#endif


/* CPU FUNCTION HEADERS*/
template<typename T>
void test_multimergesort(int p, int N);

template<typename T, fptr_t f>
void test_squareSort(int N);

/*	Main function to test either a single input size
 *	or test a whole seires of input sizes.
 */
int main(int argc, char** argv) {

	if(argc != 2 && argc != 1) {
	printf("Usage for single input size: ./testMM <N>\nUsage for multiple tests: ./testMM\n");
	exit(1);
	}


	if (argc == 2) { // SINGLE INPUT
		// Test the sorting algorithm on a single array of size N.
		int N = atoi(argv[1]);
		test_multimergesort<DATATYPE>(BLOCKS, N);
	}

	if (argc == 1) { // MULTIPLE TESTS
		// Test "nice values" ... values that fit M * K^i
		// Test values that fit M * K^i + 1 because these pad the most to reach the next "nice value"

		// 1024 * 4^1
		test_multimergesort<DATATYPE>(BLOCKS, 4096);
		test_multimergesort<DATATYPE>(BLOCKS, 4097);

		// 1024 * 4^2
		test_multimergesort<DATATYPE>(BLOCKS, 16384);
		test_multimergesort<DATATYPE>(BLOCKS, 16385);
		
		// 1024 * 4^3
		test_multimergesort<DATATYPE>(BLOCKS, 65536);
		test_multimergesort<DATATYPE>(BLOCKS, 65537);
		
		// 1024 * 4^4
		test_multimergesort<DATATYPE>(BLOCKS, 262144);
		test_multimergesort<DATATYPE>(BLOCKS, 262145);
		
		// 1024 * 4^5
		test_multimergesort<DATATYPE>(BLOCKS, 1048576);
		test_multimergesort<DATATYPE>(BLOCKS, 1048577);

		// 1024 * 4^6
		test_multimergesort<DATATYPE>(BLOCKS, 4194304);
		test_multimergesort<DATATYPE>(BLOCKS, 4194305);
	}

	#ifdef SKIP_PADDED_PARTITION
	printf("SKIPPED PADDED PARTITION\t");
	#endif

	#ifndef SKIP_PADDED_PARTITION
	printf("DID NOT SKIP PADDED PARTITION\t");
	#endif
	
	#ifdef SKIP_PADDED_MERGE
	printf("SKIPPED PADDED MERGE\n");
	#endif
	
	#ifndef SKIP_PADDED_MERGE
	printf("DID NOT SKIP PADDED MERGE\n");
	#endif

	return 0;
}


/*	p - an integer describing the number of blocks used on the GPU
 *	N - an integer describing the size of array to be sorted
 *	This function creates a array of size N and sorts it using a
 *	multiway mergesort.
 */
template<typename T>
void test_multimergesort(int p, int N) {

	// Initialize variables used to measure the runtime of the program.
	cudaEvent_t start, stop;
	float time_elapsed=0.0;
	float minTime=99999;
	float maxTime=0.0;
	float total_time=0.0;

	// Create file streams to write out the input and output
	ofstream input_file;
	ofstream output_file;
	input_file.open("input.txt");
	output_file.open("output.txt");


	/*	Possible fix: Figure out if the array needs to be padded with extra values
	 *	so that its size will match M * K^i (only necessary for N > 1024 (M?)).
	 *	If it needs to be padded, keep the next highest value of M * K^i.
	 */
	int new_N = N; // make a new value of N for padding
	#ifdef USE_PADDING
	// Keep incrementing new_N by a factor of K until it is greater or equal to N
	new_N = M;
	while (new_N < N) {
		new_N *= K;
	}
	#endif

	#ifdef PRINT_DEBUG
	// Print the adjusted (new) value of N for comparison
	printf("N: %d\tnew_N: %d\tPadded Elements: %d\n", N, new_N, new_N - N);
	#endif

	// Allocate space for the input array on the CPU
	T* h_data = (T*)malloc(new_N*sizeof(T));

	// Allocate space for input and output arrays on the GPU
	T* d_data;
	T* d_output;
	cudaMalloc(&d_data, new_N*sizeof(T));
	// cudaMalloc(&d_output, new_N*sizeof(T));

	// srand(time(NULL)); // pseudo-random seeding using time
	srand(SEED); // consistent seeding at 11 for testing
	for(int it=0; it<ITERS; it++) {

		// TEST ARRAYS
		create_random_array<T>(h_data, N, 0); // Create random array of size N to be sorted
		// create_test1_array<T>(h_data, N); // Create array of size N with mostly duplicate values

		#ifdef USE_PADDING
		// Pad the array with (new_N - N) elements. If N == new_N, no new elements are added.
		pad_array<T>(h_data, N, new_N);
		#endif

		#ifdef PRINT_DEBUG
		// Print the unsorted, padded array
		if (it == ITERS - 1) {
			printf("UNSORTED ARRAY\n");
			for (int i = 0; i < new_N; i++) {
				printf("%d\t%d\n", i, h_data[i]);
				input_file << i << "\t" << h_data[i] << "\n";
			}
			printf("\n");
		}
		#endif

		// Copy array to GPU
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

		// Record the time it took to sort the array
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
	// Calculate the average time to sort an array.
	float average_time = total_time/ITERS;
	printf("Sorted %d input(s) of size %d\nTotal Time: %lf\nAverage Time: %lf\nMin Time: %lf\nMax Time: %lf\n", ITERS, N, total_time, average_time, minTime, maxTime);

	// copy sorted result back to CPU
	cudaMemcpy(h_data, d_output, N*sizeof(T), cudaMemcpyDeviceToHost);

	// If output debug mode is on, check that output is correct
#ifdef OUTPUT_DEBUG
	/*	Loop through the array and look for any adjacent elements that
	 *	are not sorted in increasing order. Print these errors.
	 *	Note: this only checks the last array if there are multiple (ITERS > 1).
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

	// Print whether or not the array is sorted
	if(error) {
		printf("NOT SORTED!\n");
	}
	else {
		#ifdef PRINT_DEBUG
		// If the array is sorted, print it out.
		for (int i = 0; i < N; i++) {
			printf("%d\t%d\n", i, h_data[i]);
			output_file << i << "\t" << h_data[i] << "\n";
		}
		#endif
		printf("SORTED!\n");
	}
#endif

	// Free dynamically allocated memory
	cudaFree(d_data);
	cudaFree(d_output);
	free(h_data);

	// Close file streams
	input_file.close();
	output_file.close();

}


// Function to test just the basecase method
template<typename T, fptr_t f>
void test_squareSort(int N) {

	cudaEvent_t start, stop;
	float time_elapsed=0.0;
	float total_time=0.0;

	T* h_data = (T*)malloc(N*sizeof(T));
	T* d_data;
	cudaMalloc(&d_data, N*sizeof(T));
	srand(time(NULL));

	for(int it=0; it < ITERS; it++) {
		create_random_array<T>(h_data, N, 0);

		cudaMemcpy(d_data, h_data, N*sizeof(T), cudaMemcpyHostToDevice);

		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);

		squareSort<T,f><<<BLOCKS,THREADS>>>(d_data, N);
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
			if(host_cmp(h_data[i+j], h_data[j+i-1]))
				sorted=false;
		} 
	}
	if(!sorted) {
		printf("NOT SORTED\n");
	}
	else {
		printf("SORTED\n");
	}

}
