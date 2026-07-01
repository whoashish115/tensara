/*
========================================
Problem  : Tanh
URL      : https://tensara.org/problems/tanh
Code     : https://github.com/whoashish115/tensara
Author   : Ashish Kumar
Language : C++23
========================================
*/

#include <cuda_runtime.h>
#include <math.h>

__global__ void tanh_kernel(
    const float* input,
    float* output,
    size_t total)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= total) return;

    output[i] = __tanhf(input[i]);
}

extern "C" void solution(
    const float* input,
    float* output,
    size_t n,
    size_t m)
{
    size_t total = n * m;

    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    tanh_kernel<<<blocks, threads>>>(
        input,
        output,
        total
    );
}