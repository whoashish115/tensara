/*
========================================
Problem  : ReLU
URL      : https://tensara.org/problems/relu
Code     : https://github.com/whoashish115/tensara
Author   : Ashish Kumar
Language : C++23
========================================
*/

#include <cuda_runtime.h>

__global__ void relu_kernel(
    const float* A,
    float* C,
    size_t total)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= total) return;

    C[i] = (A[i] > 0.0f) ? A[i] : 0.0f;
}

extern "C" void solution(
    const float* A,
    float* C,
    size_t M,
    size_t N)
{
    size_t total = M * N;

    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    relu_kernel<<<blocks, threads>>>(A, C, total);
}