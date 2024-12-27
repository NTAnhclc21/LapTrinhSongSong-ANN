#include "tiled_kernel.h"
#include <cuda_runtime.h>

__global__ void tiledSharedMemoryKernel(float* input, float* weights, float* output, int width, int height) {
    __shared__ float tileInput[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float tileWeights[BLOCK_SIZE][BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + tx;
    int row = blockIdx.y * BLOCK_SIZE + ty;

    float sum = 0.0f;

    for (int t = 0; t < (width + BLOCK_SIZE - 1) / BLOCK_SIZE; t++) {
        if (row < height && t * BLOCK_SIZE + tx < width)
            tileInput[ty][tx] = input[row * width + t * BLOCK_SIZE + tx];
        else
            tileInput[ty][tx] = 0.0f;

        if (col < width && t * BLOCK_SIZE + ty < height)
            tileWeights[ty][tx] = weights[(t * BLOCK_SIZE + ty) * width + col];
        else
            tileWeights[ty][tx] = 0.0f;

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; k++)
            sum += tileInput[ty][k] * tileWeights[k][tx];

        __syncthreads();
    }

    if (row < height && col < width)
        output[row * width + col] = sum;
}


