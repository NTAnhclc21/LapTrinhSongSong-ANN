#include "neural_network.h"
#include "mnist_file.h"
#include "hyperparameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>


double activation_time = 0.0;
double relu_time = 0.0;
double softmax_time = 0.0;
double error_time = 0.0;
double gradient_time = 0.0;
double update_time = 0.0;
double training_time = 0.0;


int main() {

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
    clock_t start, end;

    start = clock();
    int batches = train_dataset->size / BATCH_SIZE;
    for (int i = 0; i < STEPS; i++) {
        // Initialise a new batch
        mnist_batch(train_dataset, &batch, BATCH_SIZE, i % batches);

        float total_loss = neural_network_training_step(&batch, &network, LEARNING_RATE);
        float accuracy = calculate_accuracy(test_dataset, &network);

        if (i % 25 == 0) {
            printf("Step %04d\tAverage Loss: %.2f\tAccuracy: %.3f\n", i, total_loss / BATCH_SIZE, accuracy);
        }
    }
    end = clock();
    training_time = ((double) (end - start)) / CLOCKS_PER_SEC;

    printf("Training completed.\n");
    printf("Time for activation calculation: %f seconds\n", activation_time);
    printf("Time for ReLU calculation: %f seconds\n", relu_time);
    printf("Time for softmax calculation: %f seconds\n", softmax_time);
    printf("Time for error calculation: %f seconds\n", error_time);
    printf("Time for gradient calculation: %f seconds\n", gradient_time);
    printf("Time for weight update: %f seconds\n", update_time);
    printf("Total training time: %f seconds\n", training_time);

    // Cleanup
    mnist_free_dataset(train_dataset);
    mnist_free_dataset(test_dataset);

    return 0;
}
