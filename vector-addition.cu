/*
========================================
Problem  : Vector Addition
URL      : https://tensara.org/problems/vector-addition
Code     : https://github.com/whoashish115/tensara
Author   : Ashish Kumar
Language : C++23
========================================
*/

#include <cuda_runtime.h>

__global__ void vec_add(
   const float* d_input1, const float* d_input2, float* d_output, size_t n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
   d_output[i] =d_input1[i] + d_input2[i];
}

extern "C" void solution(const float* d_input1, const float* d_input2, float* d_output, size_t n) {
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    vec_add<<<blocks, threads>>>(d_input1, d_input2, d_output, n);
}