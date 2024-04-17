#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/random.h>

#define RANGE 1048576

template<typename T>
void create_random_list(T* data, int size);

struct IntegerComparator {
    __host__ __device__ bool operator()(int a, int b) const {
        return a*a-1 <= b*b+2;
    }
};

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

int main(int argc, char ** argv) {
    cudaEvent_t start, stop;
    float time_elapsed = 0.0;

    if (argc != 2) {
      printf("Usage: ./thrust <N>\n");
    }
    int N = atoi(argv[1]);
    // Host vector of integers
    int* h_array = (int*) malloc(N * sizeof(int));

    create_random_list<int>(h_array, N);

    // Copy the host array to device
    thrust::device_vector<int> d_vec(h_array, h_array + N);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Sort on device using the custom comparator
    thrust::sort(d_vec.begin(), d_vec.end(), IntegerComparator());

    cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time_elapsed, start, stop);

  printf("%f\n", time_elapsed);

    // Copy the sorted device vector back to host
    // thrust::copy(d_vec.begin(), d_vec.end(), h_array);
    

    return 0;
}