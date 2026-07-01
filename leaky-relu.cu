/*
========================================
Problem  : Leaky ReLU
URL      : https://tensara.org/problems/leaky-relu
Code     : https://github.com/whoashish115/tensara
Author   : Ashish Kumar
Language : C++23
========================================
*/

#include <cuda_runtime.h>

__global__ void leaky_relu_kernel(
    const float* input,
    float alpha,
    float* output,
    size_t total)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= total) return;

    float x = input[i];

    output[i] = (x > 0.0f) ? x : alpha * x;
}

extern "C" void solution(
    const float* input,
    float alpha,
    float* output,
    size_t n,
    size_t m)
{
    size_t total = n * m;

    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    leaky_relu_kernel<<<blocks, threads>>>(
        input,
        alpha,
        output,
        total
    );
}