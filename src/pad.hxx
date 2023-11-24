template <typename T>
__global__ void pad(T* data, T maximum){
    *(data+threadIdx.x) = maximum;
}