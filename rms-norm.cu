/*
========================================
Problem  : RMS Normalization
URL      : https://tensara.org/problems/rms-norm
Code     : https://github.com/whoashish115/tensara
Author   : Ashish Kumar
Language : C++23
========================================
*/

#include <cuda_runtime.h>
#include <math.h>

__global__ void rmsnorm_kernel(
    const float* X,
    float* Y,
    size_t N)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;

    __shared__ float ssum[256];
    float local_sum = 0.0f;
    for (size_t col = tid; col < N; col += blockDim.x) {
        float x = X[row * N + col];
        local_sum += x * x;
    }

    ssum[tid] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            ssum[tid] += ssum[tid + stride];
        }
        __syncthreads();
    }

    float rms = sqrtf(ssum[0] / (float)N + 1e-5f);
    for (size_t col = tid; col < N; col += blockDim.x) {
        Y[row * N + col] = X[row * N + col] / rms;
    }
}
extern "C" void solution(
    const float* X,
    float* Y,
    size_t B,
    size_t N)
{
    const int threads = 256;

    rmsnorm_kernel<<<B, threads>>>(
        X,
        Y,
        N
    );
}