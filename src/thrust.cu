#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/random.h>

#define RANGE 1048576

template<typename T>
void create_random_list(T* data, int size);

int main(int argc, char* argv[]) {
  cudaEvent_t start, stop;
  int* d_data; 
  int* h_data;
  float time_elapsed = 0.0;

  if (argc != 2) {
    printf("Usage: ./thrust <N>\n");
  }
  int N = atoi(argv[1]);

  h_data = (int*) malloc(N * sizeof(int));
  create_random_list<int>(h_data, N);
  cudaMalloc(&d_data, N * sizeof(int));
  
  cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // Sort data on the device.
  thrust::sort(thrust::device, d_data, d_data + N);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time_elapsed, start, stop);

  printf("%f\n", time_elapsed);
}

template<typename T>
void create_random_list(T* data, int size) {
  long temp;
//printf("size:%d\n", size);
  for(int i=0; i<size; i++) {
//    data[i].key = rand()%RANGE + min;
//    data[i].val = rand()%RANGE + min;
    data[i] = (rand()%RANGE);
    temp = rand()%RANGE;
    data[i] += (temp<<32);
  }
}