/*
========================================
Problem  : 1D Convolution
URL      : https://tensara.org/problems/conv-1d
Code     : https://github.com/whoashish115/tensara
Author   : Ashish Kumar
Language : C++23
========================================
*/

#include <cuda_runtime.h>

__global__ void conv1d_kernel(
    const float* A,
    const float* B,
    float* C,
    size_t N,
    size_t K)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float sum = 0.0f;
    int radius = (int)K / 2;

    for (int j = -radius; j <= radius; j++) {
        int idx = i + j;

        if (idx >= 0 && idx < (int)N) {
            sum += A[idx] * B[j + radius];
        }
    }

    C[i] = sum;
}

extern "C" void solution(
    const float* A,
    const float* B,
    float* C,
    size_t N,
    size_t K)
{
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;
    conv1d_kernel<<<blocks, threads>>>(A, B, C, N, K);
}