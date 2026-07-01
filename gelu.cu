/*
========================================
Problem  : GELU
URL      : https://tensara.org/problems/gelu
Code     : https://github.com/whoashish115/tensara
Author   : Ashish Kumar
Language : C++23
========================================
*/

#include <cuda_runtime.h>
#include <math.h>

__global__ void gelu_kernel(
    const float* input,
    float* output,
    size_t total)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    float x = input[i];
    const float kAlpha = 0.7978845608f; // sqrt(2/pi)
    float x3 = x * x * x;
    output[i] =
        0.5f * x *
        (1.0f + tanhf(
            kAlpha * (x + 0.044715f * x3)
        ));
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

    gelu_kernel<<<blocks, threads>>>(
        input,
        output,
        total
    );
}