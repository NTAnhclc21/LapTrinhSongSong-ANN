#include "neural_network.h"
#include "mnist_file.h"
#include "hyperparameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

    printf("Training completed.\n");

    // Cleanup
    mnist_free_dataset(train_dataset);
    mnist_free_dataset(test_dataset);

    return 0;
}
