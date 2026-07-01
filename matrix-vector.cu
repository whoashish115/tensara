/*
========================================
Problem  : Matrix Vector Multiplication
URL      : https://tensara.org/problems/matrix-vector
Code     : https://github.com/whoashish115/tensara
Author   : Ashish Kumar
Language : C++23
========================================
*/

#include <cuda_runtime.h>

__global__ void matvec_kernel(
    const float* A,
    const float* B,
    float* C,
    size_t M,
    size_t K)
{
    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;
    float sum = 0.0f;
    for (size_t j = 0; j < K; ++j) {
        sum += A[row * K + j] * B[j];
    }

    C[row] = sum;
}

// A, B, C are device pointers
extern "C" void solution(
    const float* input_a,
    const float* input_b,
    float* output_c,
    size_t m,
    size_t k)
{
    const int threads = 256;
    const int blocks = (m + threads - 1) / threads;

    matvec_kernel<<<blocks, threads>>>(
        input_a,
        input_b,
        output_c,
        m,
        k
    );
}