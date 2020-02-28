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

#ifndef buildData_h
	#define buildData_h
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>


/* CPU Functions to create testData */
template<typename T>
void create_sorted_array(T* data, int size, int addVal) {
	for(int i=0; i<size; i++) {
		// data[i].key = i+addVal;
		// data[i].val = i-addVal;
  }
}
/*
template<typename T>
void create_random_sorted_array(T* data, int size) {
	// srand(time(NULL));
	for(int i=0; i<size; i++) {
		data[i] = {.key = rand()%RANGE, .val= rand()%RANGE};
	}
	std::sort(data, data+size);
}
*/

template<typename T>
void create_random_array(T* data, int size, int min) {
	// srand(time(NULL)); // pseudo-random seeding using time
	// srand(56); // consistent seeding for testing
	// long temp;
		// printf("size:%d\n", size);
	for(int i=0; i<size; i++) {
	// data[i].key = rand()%RANGE + min;
		// data[i].val = rand()%RANGE + min;
		// data[i] = (rand()%RANGE) + min + 1;
		data[i] = (rand() % RANGE) + min + 1;
		// temp = rand()%RANGE + min + 1;
		// 	data[i] += (temp<<32);
	}
}
#endif


/*	create_test1_array makes an array with almost all of the
 *	same value to evaluate effect on partitioning.
 */
template<typename T>
void create_test1_array(T * data, int size) {
	for (int i = 0; i < size-2; i++) {
		data[i] = RANGE/2;
	}
	data[size-2] = RANGE;
	data[size-1] = RANGE;
}


/*	pad_array takes in an array of "current_size" elements and appends
 *	values greater than or equal to the largest value in the
 *	existing array until there are "new_size" elements in total
 */
template<typename T>
void pad_array(T * data, int current_size, int new_size) {
	// iterates from the index of current_size and fills up the rest of the array
	for (int i = current_size; i < new_size; i++) {
		data[i] = RANGE + 1;
	}
}