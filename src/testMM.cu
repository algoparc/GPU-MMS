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

	cudaEvent_t start, stop;
	float time_elapsed=0.0;
	float minTime=99999;
	float maxTime=0.0;

	// Create sample sorted lists
	T* h_data = (T*)malloc(N*sizeof(T));

	T* d_data;
	T* d_output;
	cudaMalloc(&d_data, N*sizeof(T));
	cudaMalloc(&d_output, N*sizeof(T));
	float total_time=0.0;

	// srand(time(NULL));
	srand(11); // consistent seeding
	for(int it=0; it<ITERS; it++) {

		// Create random list to be sorted
		create_random_list<T>(h_data, N, 0);

		// for (int i = 0; i < N; i++) {
		// 	printf("%d\n", h_data[i]);
		// }

		// Copy list to GPU
		cudaMemcpy(d_data, h_data, N*sizeof(T), cudaMemcpyHostToDevice);

		// Zero out result array
		cudaMalloc(&d_output, N*sizeof(T));
		//  cudaMemset(&d_output, 0, N*sizeof(T));

		cudaDeviceSynchronize();
		// Timer functions
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);

		// Run GPU-MMS.  T is datatype and cmp is comparison function (defined in cmp.hxx)
		d_output = multimergesort<T,cmp>(d_data, d_output, h_data, p, N);
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
	total_time = total_time/ITERS;
	printf("%lf %lf %lf\n", total_time, minTime, maxTime);

	// copy sorted result back to CPU
	cudaMemcpy(h_data, d_output, N*sizeof(T), cudaMemcpyDeviceToHost);

	// If debug mode is on, check that output is correct
#ifdef DEBUG

	// print the sorted array
	// for (int i = 0; i < N; i++) {
	// 	printf("%d\t%d\n", i, h_data[i]);
	// }

	bool error=false;
	for(int i=1; i<N; i++) {
		if(host_cmp<int>(h_data[i], h_data[i-1])) {
			printf("ERROR FOUND!\n");
			printf("h_data[%d]: %d\n", i, h_data[i]);
			printf("h_data[%d]: %d\n", i-1, h_data[i-1]);
			error=true;
		}
	}
	if(error)
		printf("NOT SORTED!\n");
	else
		printf("SORTED!\n");

	// for (int i = 0; i < N; i++) {
	// 	printf("%d\n", h_data[i]);
	// }

#endif

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
