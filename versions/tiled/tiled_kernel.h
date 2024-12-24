#pragma once
#ifndef TILED_KERNEL_H
#define TILED_KERNEL_H

#define BLOCK_SIZE 16

__global__ void tiledSharedMemoryKernel(float* input, float* weights, float* output, int width, int height);

#endif

