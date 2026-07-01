/*
========================================
Problem  : 1D Average Pooling
URL      : https://tensara.org/problems/avg-pool-1d
Code     : https://github.com/whoashish115/tensara
Author   : Ashish Kumar
Language : C++23
========================================
*/

#include <cuda_runtime.h>

__global__ void avgpool1d_kernel(
    const float* input,
    float* output,
    size_t H,
    size_t Hout,
    int kernel_size,
    int stride,
    int padding)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= Hout) return;

    int start = (int)i * stride - padding;

    float sum = 0.0f;

    for (int j = 0; j < kernel_size; ++j) {
        int idx = start + j;

        if (idx >= 0 && idx < (int)H) {
            sum += input[idx];
        }
    }

    output[i] = sum / kernel_size;
}

extern "C" void solution(
    const float* input,
    int kernel_size,
    int stride,
    int padding,
    float* output,
    size_t H)
{

    const int threads = 256;
    size_t Hout =
        (H + 2 * padding - kernel_size) / stride + 1;
    const int blocks = (Hout + threads - 1) / threads;

    avgpool1d_kernel<<<blocks, threads>>>(
        input,
        output,
        H,
        Hout,
        kernel_size,
        stride,
        padding
    );
}