#include "neural_network.h"
#include "mnist_file.h"
#include "hyperparameters.h"
#include <stdio.h>

int main() {
    mnist_dataset_t * train_dataset, * test_dataset;
    mnist_dataset_t batch;
    neural_network_t network;

    // Read the datasets from the files
    train_dataset = mnist_get_dataset(TRAIN_IMAGES, TRAIN_LABELS);
    test_dataset = mnist_get_dataset(TEST_IMAGES, TEST_LABELS);

    // Initialise weights and biases with random values
    neural_network_random_weights(&network);

    // Training loop
    int batches = train_dataset->size / BATCH_SIZE;
    for (int i = 0; i < STEPS; i++) {
        // Initialise a new batch
        mnist_batch(&train_dataset, &batch, BATCH_SIZE, i % batches);

        // Running one step (epoch) of gradient descent
        float total_loss = neural_network_training_step(&batch, &network, LEARNING_RATE);
        float accuracy = calculate_accuracy(test_dataset, &network);

        if (i % 25 == 0) {
            printf("Step %04d\tAverage Loss: %.2f\tAccuracy: %.3f\n", i, total_loss / BATCH_SIZE, accuracy);
        }
    }

    // Cleanup
    mnist_free_dataset(train_dataset);
    mnist_free_dataset(test_dataset);

    return 0;
}



#include "neural_network.h"
#include "mnist_file.h"
#include "hyperparameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Execution mode
typedef enum {
    CPU,
    GPU_V0,  // Simple GPU implementation
    GPU_V1,  // Optimized GPU implementation
    GPU_V2   // Advanced GPU optimization
} execution_mode_t;

// Function to parse command-line arguments and return execution mode
execution_mode_t get_execution_mode(const char *arg) {
    if (strcmp(arg, "cpu") == 0) return CPU;
    if (strcmp(arg, "gpu_v0") == 0) return GPU_V0;
    if (strcmp(arg, "gpu_v1") == 0) return GPU_V1;
    if (strcmp(arg, "gpu_v2") == 0) return GPU_V2;

    fprintf(stderr, "Invalid execution mode: %s\n", arg);
    fprintf(stderr, "Available modes: cpu, gpu_v0, gpu_v1, gpu_v2\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <execution_mode>\n", argv[0]);
        return EXIT_FAILURE;
    }

    execution_mode_t mode = get_execution_mode(argv[1]);

    // Initialize variables
    mnist_dataset_t * train_dataset, * test_dataset;
    mnist_dataset_t batch;
    neural_network_t network;

    // Load MNIST datasets
    train_dataset = mnist_get_dataset(TRAIN_IMAGES, TRAIN_LABELS);
    test_dataset = mnist_get_dataset(TEST_IMAGES, TEST_LABELS);

    // Initialise weights and biases with random values
    neural_network_random_weights(&network);

    // Training loop
    int batches = train_dataset->size / BATCH_SIZE;
    printf("Training started with mode: %s\n", argv[1]);

    for (int i = 0; i < STEPS; i++) {
        // Initialise a new batch
        mnist_batch(train_dataset, &batch, BATCH_SIZE, i % batches);

        float total_loss = 0.0f;

        switch (mode) {
            case CPU:
                total_loss = neural_network_training_step(&batch, &network, LEARNING_RATE);
                break;

            case GPU_V0:
                total_loss = neural_network_training_step_gpu_v0(&batch, &network, LEARNING_RATE);
                break;

            // case GPU_V1:
            //     total_loss = neural_network_training_step_gpu_v1(&batch, &network, LEARNING_RATE);
            //     break;

            // case GPU_V2:
            //     total_loss = neural_network_training_step_gpu_v2(&batch, &network, LEARNING_RATE);
            //     break;

            default:
                fprintf(stderr, "Unsupported execution mode.\n");
                exit(EXIT_FAILURE);
        }

        float accuracy = calculate_accuracy(test_dataset, &network);

        if (i % 25 == 0) {
            printf("Step %04d\tAverage Loss: %.2f\tAccuracy: %.3f\n", i, total_loss / BATCH_SIZE, accuracy);
        }
    }

    printf("Training completed.\n");

    // Cleanup
    mnist_free_dataset(train_dataset);
    mnist_free_dataset(test_dataset);

    return 0;
}
