#include "neural_network.h"
#include "mnist_file.h"
#include "hyperparameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define INPUT_LAYER_SIZE      784       // MNIST-fashionfashion 28x28 image
#define HIDDEN_LAYER1_SIZE    128
#define HIDDEN_LAYER2_SIZE    128       // 2 hidden layers
#define OUTPUT_LAYER_SIZE     10        // MNIST-fashion 10 labels

#define BATCH_SIZE            32
#define LEARNING_RATE         0.5f
#define STEPS                 500    

// Global variable definitions
double training_time = 0.0;

int main() {
    // Initialize variables
    mnist_dataset_t *train_dataset, *test_dataset;
    mnist_dataset_t batch;
    neural_network_t network;

    // Dynamically allocate memory for weights and biases
    float* W1 = (float*)malloc(INPUT_LAYER_SIZE * HIDDEN_LAYER1_SIZE * sizeof(float));
    float* b1 = (float*)malloc(HIDDEN_LAYER1_SIZE * sizeof(float));

    float* W2 = (float*)malloc(HIDDEN_LAYER1_SIZE * HIDDEN_LAYER2_SIZE * sizeof(float));
    float* b2 = (float*)malloc(HIDDEN_LAYER2_SIZE * sizeof(float));

    float* W3 = (float*)malloc(HIDDEN_LAYER2_SIZE * OUTPUT_LAYER_SIZE * sizeof(float));
    float* b3 = (float*)malloc(OUTPUT_LAYER_SIZE * sizeof(float));

// Check if memory allocation was successful
    if (W1 == NULL || b1 == NULL || W2 == NULL || b2 == NULL || W3 == NULL || b3 == NULL) {
       printf("Memory allocation failed.\n");
       // Handle memory allocation failure
       return -1; // Exit the program or handle it as needed
    }

// Assign dynamically allocated memory to the network struct
    memcpy(network.W1, W1, INPUT_LAYER_SIZE * HIDDEN_LAYER1_SIZE * sizeof(float));
    memcpy(network.b1, b1, HIDDEN_LAYER1_SIZE * sizeof(float));

    memcpy(network.W2, W2, HIDDEN_LAYER1_SIZE * HIDDEN_LAYER2_SIZE * sizeof(float));
    memcpy(network.b2, b2, HIDDEN_LAYER2_SIZE * sizeof(float));

    memcpy(network.W3, W3, HIDDEN_LAYER2_SIZE * OUTPUT_LAYER_SIZE * sizeof(float));
    memcpy(network.b3, b3, OUTPUT_LAYER_SIZE * sizeof(float));

    // Load MNIST datasets
    train_dataset = mnist_get_dataset(TRAIN_IMAGES, TRAIN_LABELS);
    test_dataset = mnist_get_dataset(TEST_IMAGES, TEST_LABELS);

    // Initialise weights and biases with random values
    neural_network_random_weights(&network);

    // Allocate memory for CUDA
    // CUDA device memory for input, weights, and output
    float* d_input;
    float* d_W1;
    float* d_b1;
    float* d_W2;
    float* d_b2;
    float* d_W3;
    float* d_b3;
    float* d_output;
    int batches = train_dataset->size / BATCH_SIZE;

    cudaMalloc(&d_input, BATCH_SIZE * 784 * sizeof(float));
    cudaMalloc(&d_W1, 784 * 128 * sizeof(float));
    cudaMalloc(&d_b1, 128 * sizeof(float));
    cudaMalloc(&d_W2, 128 * 128 * sizeof(float));
    cudaMalloc(&d_b2, 128 * sizeof(float));
    cudaMalloc(&d_W3, 128 * 10 * sizeof(float));
    cudaMalloc(&d_b3, 10 * sizeof(float));
    cudaMalloc(&d_output, BATCH_SIZE * 10 * sizeof(float));

    cudaMemcpy(d_W1, network.W1, 784 * 128 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1, network.b1, 128 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, network.W2, 128 * 128 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2, network.b2, 128 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W3, network.W3, 128 * 10 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b3, network.b3, 10 * sizeof(float), cudaMemcpyHostToDevice);

    // Training loop
    clock_t start, end;
    start = clock();

    //int batches = train_dataset->size / BATCH_SIZE;
    for (int i = 0; i < STEPS; i++) {
        // Initialise a new batch
        mnist_batch(train_dataset, &batch, BATCH_SIZE, i % batches);

        // Copy batch input to device
        cudaMemcpy(d_input, batch.images, BATCH_SIZE * 784 * sizeof(float), cudaMemcpyHostToDevice);


        // Launch Tiled Shared Memory kernel for activation
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid((784 + BLOCK_SIZE - 1) / BLOCK_SIZE, (HIDDEN_LAYER1_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE);

        tiledSharedMemoryKernel<<<grid, block>>>(d_input, d_W1, d_output, 784, HIDDEN_LAYER1_SIZE);

        // Perform one step of training (forward pass, loss, backward pass, etc.)
        float total_loss = neural_network_training_step(&batch, &network, LEARNING_RATE);

        // Calculate accuracy
        float accuracy = calculate_accuracy(test_dataset, &network);

        // Print metrics every 25 steps
        if (i % 25 == 0) {
            printf("Step %04d\tAverage Loss: %.2f\tAccuracy: %.3f\n", i, total_loss / BATCH_SIZE, accuracy);
        }
    }

    end = clock();
    training_time = ((double)(end - start)) / CLOCKS_PER_SEC;

    printf("Training completed.\n\n");
    printf("Total training time: %f seconds\n", training_time);

    // Cleanup
    mnist_free_dataset(train_dataset);
    mnist_free_dataset(test_dataset);
    cudaFree(d_input);
    cudaFree(d_W1);
    cudaFree(d_b1);
    cudaFree(d_W2);
    cudaFree(d_b2);
    cudaFree(d_W3);
    cudaFree(d_b3);
    cudaFree(d_output);

    return 0;
}

