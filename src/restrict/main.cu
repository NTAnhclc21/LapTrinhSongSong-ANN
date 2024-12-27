#include "neural_network.h"
#include "mnist_file.h"
#include "hyperparameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <float.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16
#define NUM_CHANNELS 1
#define INPUT_WIDTH 28
#define INPUT_HEIGHT 28
#define BATCH_SIZE 32
#define STEPS 500
#define LEARNING_RATE 0.5f

double reduction_time = 0.0;
double activation_time = 0.0;
double training_time = 0.0;

// Kernel restrict and loop unrolling
__global__ void optimizedKernel(float* __restrict__ d_input, float* __restrict__ d_output, 
                                 int numChannels, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx >= width || idy >= height) return;

    float result = 0.0f;

    // Unrolling loop
    for (int i = 0; i < 4; i++) {
        if (idx + i < width && idy + i < height) {
            result += d_input[(idy + i) * width + (idx + i)];
        }
    }

    d_output[idy * width + idx] = result;
}

// Tune and Train
void tuneAndTrain(mnist_dataset_t *train_dataset, mnist_dataset_t *test_dataset, neural_network_t *network) {

    float *d_input, *d_output;
    int numChannels = NUM_CHANNELS;
    int width = INPUT_WIDTH;
    int height = INPUT_HEIGHT;
    
    int inputSize = BATCH_SIZE * numChannels * width * height * sizeof(float);
    int outputSize = BATCH_SIZE * width * height * sizeof(float);
    cudaMalloc(&d_input, inputSize);
    cudaMalloc(&d_output, outputSize);

    // Different blocksizes
    int blockSizes[] = {16, 32, 64}; 
    int bestBlockSize = 16;
    float bestTime = FLT_MAX;
    
    for (int blockSizeIndex = 0; blockSizeIndex < 3; blockSizeIndex++) {
        int blockSize = blockSizes[blockSizeIndex];
        dim3 block(blockSize, blockSize);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
        
        clock_t start, end;
        start = clock();
        
        int batches = train_dataset->size / BATCH_SIZE;
        for (int step = 0; step < STEPS; step++) {
            mnist_dataset_t batch; 
            
            // Load batch from train_dataset
            mnist_batch(train_dataset, &batch, BATCH_SIZE, step % batches);
            
            // Copy batch input to device
            cudaMemcpy(d_input, batch.images, BATCH_SIZE * numChannels * width * height * sizeof(float), cudaMemcpyHostToDevice);

            optimizedKernel<<<grid, block>>>(d_input, d_output, numChannels, width, height);
            
            // Check CUDA error
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
                return;
            }

            float total_loss = neural_network_training_step(&batch, network, LEARNING_RATE);
            float accuracy = calculate_accuracy(test_dataset, network);

            if (step % 25 == 0) {
                printf("Step %04d\tAverage Loss: %.2f\tAccuracy: %.3f\n", step, total_loss / BATCH_SIZE, accuracy);
            }
        }

        end = clock();
        float elapsedTime = ((float)(end - start)) / CLOCKS_PER_SEC;
        printf("Time for block size %d: %.3f seconds\n", blockSize, elapsedTime);
        
        // Best time
        if (elapsedTime < bestTime) {
            bestTime = elapsedTime;
            bestBlockSize = blockSize;
        }
    }

    printf("Best block size: %d\n", bestBlockSize);
    
    // Free
    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    mnist_dataset_t *train_dataset, *test_dataset;
    neural_network_t network;

    train_dataset = mnist_get_dataset(TRAIN_IMAGES, TRAIN_LABELS);
    test_dataset = mnist_get_dataset(TEST_IMAGES, TEST_LABELS);

    neural_network_random_weights(&network);

    tuneAndTrain(train_dataset, test_dataset, &network);

    mnist_free_dataset(train_dataset);
    mnist_free_dataset(test_dataset);

    return 0;
}


