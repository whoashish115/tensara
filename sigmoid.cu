/*
========================================
Problem  : Sigmoid
URL      : https://tensara.org/problems/sigmoid
Code     : https://github.com/whoashish115/tensara
Author   : Ashish Kumar
Language : C++23
========================================
*/

#include <cuda_runtime.h>

__global__ void sigmoid_kernel(
    const float* input,
    float* output,
    size_t total)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < total) {
        float x = input[i];
        output[i] = 1.0f / (1.0f + __expf(-x));
    }
}

extern "C" void solution(
    const float* input,
    float* output,
    size_t n,
    size_t m)
{
    size_t total = n * m;

    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    sigmoid_kernel<<<blocks, threads>>>(
        input,
        output,
        total
    );
}