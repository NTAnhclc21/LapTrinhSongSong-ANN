#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include <cuda_fp16.h>

#include "neural_network.h"
#include "mnist_file.h"
#include "hyperparameters.h"

#define INPUT_LAYER_SIZE 784
#define HIDDEN_LAYER1_SIZE 128
#define HIDDEN_LAYER2_SIZE 128
#define OUTPUT_LAYER_SIZE 10
#define BATCH_SIZE 32
#define STEPS 500
#define LEARNING_RATE 0.5f
#define BLOCK_SIZE 16

__global__ void activation_kernel(__half* d_input, __half* d_output, int input_size, int output_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < input_size && y < output_size) {
        // Flattening input into 2D structure (assuming a simple fully connected layer)
        int idx = y * input_size + x;

        // Apply ReLU activation: max(0, input)
        d_output[idx] = __hmax(__half(0.0f), d_input[idx]);
    }
}

// Main function
int main() {
    // Initialize variables
    mnist_dataset_t *train_dataset, *test_dataset;
    mnist_dataset_t batch;
    neural_network_t network;

    // Dynamically allocate memory for weights and biases using FP16
    __half* W1 = (__half*)malloc(INPUT_LAYER_SIZE * HIDDEN_LAYER1_SIZE * sizeof(__half));
    __half* b1 = (__half*)malloc(HIDDEN_LAYER1_SIZE * sizeof(__half));
    __half* W2 = (__half*)malloc(HIDDEN_LAYER1_SIZE * HIDDEN_LAYER2_SIZE * sizeof(__half));
    __half* b2 = (__half*)malloc(HIDDEN_LAYER2_SIZE * sizeof(__half));
    __half* W3 = (__half*)malloc(HIDDEN_LAYER2_SIZE * OUTPUT_LAYER_SIZE * sizeof(__half));
    __half* b3 = (__half*)malloc(OUTPUT_LAYER_SIZE * sizeof(__half));

    // Check if memory allocation was successful
    if (W1 == NULL || b1 == NULL || W2 == NULL || b2 == NULL || W3 == NULL || b3 == NULL) {
        printf("Memory allocation failed.\n");
        return -1;
    }

    // Assign dynamically allocated memory to the network struct (using FP16)
    memcpy(network.W1, W1, INPUT_LAYER_SIZE * HIDDEN_LAYER1_SIZE * sizeof(__half));
    memcpy(network.b1, b1, HIDDEN_LAYER1_SIZE * sizeof(__half));
    memcpy(network.W2, W2, HIDDEN_LAYER1_SIZE * HIDDEN_LAYER2_SIZE * sizeof(__half));
    memcpy(network.b2, b2, HIDDEN_LAYER2_SIZE * sizeof(__half));
    memcpy(network.W3, W3, HIDDEN_LAYER2_SIZE * OUTPUT_LAYER_SIZE * sizeof(__half));
    memcpy(network.b3, b3, OUTPUT_LAYER_SIZE * sizeof(__half));

    // Load MNIST datasets
    train_dataset = mnist_get_dataset(TRAIN_IMAGES, TRAIN_LABELS);
    test_dataset = mnist_get_dataset(TEST_IMAGES, TEST_LABELS);

    // Initialize weights and biases with random values (using FP16)
    neural_network_random_weights(&network);

    // Allocate memory for CUDA using FP16
    __half* d_input;
    __half* d_output;

    int batches = train_dataset->size / BATCH_SIZE;

    cudaMalloc(&d_input, BATCH_SIZE * 784 * sizeof(__half));
    cudaMalloc(&d_output, BATCH_SIZE * 10 * sizeof(__half));

    // Training loop
    clock_t start, end;
    start = clock();

    for (int i = 0; i < STEPS; i++) {
        // Initialize a new batch
        mnist_batch(train_dataset, &batch, BATCH_SIZE, i % batches);

        // Copy batch input to device (convert from float to FP16)
        __half* h_input_fp16 = (__half*)malloc(BATCH_SIZE * 784 * sizeof(__half));
        for (int j = 0; j < MNIST_IMAGE_SIZE; ++j) {
        // Convert pixel value from uint8_t to float (normalize if needed, e.g., divide by 255)
        float pixel_value = (float)batch.images[j].pixels[j] / 255.0f;  // Normalizing to [0, 1]

        // Convert the float value to FP16
        h_input_fp16[j] = __float2half(pixel_value);  // Convert float to FP16
    }

        cudaMemcpy(d_input, h_input_fp16, BATCH_SIZE * 784 * sizeof(__half), cudaMemcpyHostToDevice);

        // Launch the kernel for activation calculation
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid((784 + BLOCK_SIZE - 1) / BLOCK_SIZE, (HIDDEN_LAYER1_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE);

        activation_kernel<<<grid, block>>>(d_input, d_output, 784, HIDDEN_LAYER1_SIZE);

        // Perform one step of training (forward pass, loss, backward pass, etc.)
        __half total_loss = neural_network_training_step(&batch, &network, LEARNING_RATE);

        // Calculate accuracy (in FP16)
        float accuracy = calculate_accuracy(test_dataset, &network);

        // Print metrics every 25 steps
        if (i % 25 == 0) {
            printf("Step %04d\tAverage Loss: %.2f\tAccuracy: %.3f\n", i, __half2float(total_loss) / BATCH_SIZE, accuracy);
        }
    }

    end = clock();
    double training_time = ((double)(end - start)) / CLOCKS_PER_SEC;

    printf("Training completed.\n\n");
    printf("Total training time: %f seconds\n", training_time);

    // Cleanup
    mnist_free_dataset(train_dataset);
    mnist_free_dataset(test_dataset);
    cudaFree(d_input);
    cudaFree(d_output);
    free(W1);
    free(b1);
    free(W2);
    free(b2);
    free(W3);
    free(b3);

    return 0;
}



