#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "include/mnist_file.h"
#include "include/neural_network.h"

#define PIXEL_SCALE(x) (((float) (x)) / 255.0f)
#define RAND_FLOAT() (((float) rand()) / ((float) RAND_MAX))

// ReLU Activation Function
float neural_network_relu(float x) {
    return x > 0 ? x : 0;
}

// ReLU Derivative (for backpropagation)
float neural_network_relu_derivative(float x) {
    return x > 0 ? 1.0f : 0.0f;
}

// Initialize weights randomly
void neural_network_random_weights(neural_network_t * network) {
    int i, j;

    // First layer weights
    for (i = 0; i < HIDDEN_LAYER1_SIZE; i++) {
        network->b1[i] = RAND_FLOAT() - 0.5f;
        for (j = 0; j < INPUT_LAYER_SIZE; j++) {
            network->W1[i][j] = (RAND_FLOAT() - 0.5f) * 0.01f;
        }
    }

    // Second layer weights
    for (i = 0; i < HIDDEN_LAYER2_SIZE; i++) {
        network->b2[i] = RAND_FLOAT() - 0.5f;
        for (j = 0; j < HIDDEN_LAYER1_SIZE; j++) {
            network->W2[i][j] = (RAND_FLOAT() - 0.5f) * 0.01f;
        }
    }

    // Output layer weights
    for (i = 0; i < OUTPUT_LAYER_SIZE; i++) {
        network->b3[i] = RAND_FLOAT() - 0.5f;
        for (j = 0; j < HIDDEN_LAYER2_SIZE; j++) {
            network->W3[i][j] = (RAND_FLOAT() - 0.5f) * 0.01f;
        }
    }
}

// Softmax function (same as before)
void neural_network_softmax(float * activations, int length) {
    int i;
    float sum, max;

    for (i = 1, max = activations[0]; i < length; i++) {
        if (activations[i] > max) {
            max = activations[i];
        }
    }

    for (i = 0, sum = 0; i < length; i++) {
        activations[i] = exp(activations[i] - max);
        sum += activations[i];
    }

    for (i = 0; i < length; i++) {
        activations[i] /= sum;
    }
}

// Forward propagation through the network
void neural_network_hypothesis(mnist_image_t * image, neural_network_t * network, float activations[MNIST_LABELS]) {
    float layer1_activations[HIDDEN_LAYER1_SIZE];
    float layer2_activations[HIDDEN_LAYER2_SIZE];
    int i, j;

    // First hidden layer
    for (i = 0; i < HIDDEN_LAYER1_SIZE; i++) {
        layer1_activations[i] = network->b1[i];
        for (j = 0; j < INPUT_LAYER_SIZE; j++) {
            layer1_activations[i] += network->W1[i][j] * PIXEL_SCALE(image->pixels[j]);
        }
        layer1_activations[i] = neural_network_relu(layer1_activations[i]);
    }

    // Second hidden layer
    for (i = 0; i < HIDDEN_LAYER2_SIZE; i++) {
        layer2_activations[i] = network->b2[i];
        for (j = 0; j < HIDDEN_LAYER1_SIZE; j++) {
            layer2_activations[i] += network->W2[i][j] * layer1_activations[j];
        }
        layer2_activations[i] = neural_network_relu(layer2_activations[i]);
    }

    // Output layer
    for (i = 0; i < OUTPUT_LAYER_SIZE; i++) {
        activations[i] = network->b3[i];
        for (j = 0; j < HIDDEN_LAYER2_SIZE; j++) {
            activations[i] += network->W3[i][j] * layer2_activations[j];
        }
    }

    neural_network_softmax(activations, OUTPUT_LAYER_SIZE);
}

// Gradient descent and update
float neural_network_gradient_update(mnist_image_t * image, neural_network_t * network, neural_network_gradient_t * gradient, uint8_t label) {
    float layer1_activations[HIDDEN_LAYER1_SIZE];
    float layer2_activations[HIDDEN_LAYER2_SIZE];
    float output_activations[OUTPUT_LAYER_SIZE];
    float layer2_errors[HIDDEN_LAYER2_SIZE];
    float layer1_errors[HIDDEN_LAYER1_SIZE];
    float output_errors[OUTPUT_LAYER_SIZE];
    int i, j;

    // Forward pass (similar to hypothesis function)
    for (i = 0; i < HIDDEN_LAYER1_SIZE; i++) {
        layer1_activations[i] = network->b1[i];
        for (j = 0; j < INPUT_LAYER_SIZE; j++) {
            layer1_activations[i] += network->W1[i][j] * PIXEL_SCALE(image->pixels[j]);
        }
        layer1_activations[i] = neural_network_relu(layer1_activations[i]);
    }

    for (i = 0; i < HIDDEN_LAYER2_SIZE; i++) {
        layer2_activations[i] = network->b2[i];
        for (j = 0; j < HIDDEN_LAYER1_SIZE; j++) {
            layer2_activations[i] += network->W2[i][j] * layer1_activations[j];
        }
        layer2_activations[i] = neural_network_relu(layer2_activations[i]);
    }

    for (i = 0; i < OUTPUT_LAYER_SIZE; i++) {
        output_activations[i] = network->b3[i];
        for (j = 0; j < HIDDEN_LAYER2_SIZE; j++) {
            output_activations[i] += network->W3[i][j] * layer2_activations[j];
        }
    }

    neural_network_softmax(output_activations, OUTPUT_LAYER_SIZE);

    // Backpropagation
    // Output layer error
    for (i = 0; i < OUTPUT_LAYER_SIZE; i++) {
        output_errors[i] = (i == label) ? output_activations[i] - 1 : output_activations[i];
    }

    // Gradient calculation for output layer
    for (i = 0; i < OUTPUT_LAYER_SIZE; i++) {
        for (j = 0; j < HIDDEN_LAYER2_SIZE; j++) {
            gradient->W3_grad[i][j] += output_errors[i] * layer2_activations[j];
        }
        gradient->b3_grad[i] += output_errors[i];
    }

    // Second hidden layer error
    for (i = 0; i < HIDDEN_LAYER2_SIZE; i++) {
        layer2_errors[i] = 0;
        for (j = 0; j < OUTPUT_LAYER_SIZE; j++) {
            layer2_errors[i] += output_errors[j] * network->W3[j][i];
        }
        layer2_errors[i] *= neural_network_relu_derivative(layer2_activations[i]);
    }

    // Second hidden layer gradient
    for (i = 0; i < HIDDEN_LAYER2_SIZE; i++) {
        for (j = 0; j < HIDDEN_LAYER1_SIZE; j++) {
            gradient->W2_grad[i][j] += layer2_errors[i] * layer1_activations[j];
        }
        gradient->b2_grad[i] += layer2_errors[i];
    }

    // First hidden layer error
    for (i = 0; i < HIDDEN_LAYER1_SIZE; i++) {
        layer1_errors[i] = 0;
        for (j = 0; j < HIDDEN_LAYER2_SIZE; j++) {
            layer1_errors[i] += layer2_errors[j] * network->W2[j][i];
        }
        layer1_errors[i] *= neural_network_relu_derivative(layer1_activations[i]);
    }

    // First hidden layer gradient
    for (i = 0; i < HIDDEN_LAYER1_SIZE; i++) {
        for (j = 0; j < INPUT_LAYER_SIZE; j++) {
            gradient->W1_grad[i][j] += layer1_errors[i] * PIXEL_SCALE(image->pixels[j]);
        }
        gradient->b1_grad[i] += layer1_errors[i];
    }

    // Cross-entropy loss for the output
    return 0.0f - log(output_activations[label]);
}

// Training step remains largely the same, but with updated gradient descent
float neural_network_training_step(mnist_dataset_t * dataset, neural_network_t * network, float learning_rate) {
    neural_network_gradient_t gradient;
    float total_loss;
    int i, j;

    memset(&gradient, 0, sizeof(neural_network_gradient_t));

    for (i = 0, total_loss = 0; i < dataset->size; i++) {
        total_loss += neural_network_gradient_update(&dataset->images[i], network, &gradient, dataset->labels[i]);
    }

    // Update weights and biases
    for (i = 0; i < HIDDEN_LAYER1_SIZE; i++) {
        network->b1[i] -= learning_rate * gradient.b1_grad[i] / dataset->size;
        for (j = 0; j < INPUT_LAYER_SIZE; j++) {
            network->W1[i][j] -= learning_rate * gradient.W1_grad[i][j] / dataset->size;
        }
    }

    for (i = 0; i < HIDDEN_LAYER2_SIZE; i++) {
        network->b2[i] -= learning_rate * gradient.b2_grad[i] / dataset->size;
        for (j = 0; j < HIDDEN_LAYER1_SIZE; j++) {
            network->W2[i][j] -= learning_rate * gradient.W2_grad[i][j] / dataset->size;
        }
    }

    for (i = 0; i < OUTPUT_LAYER_SIZE; i++) {
        network->b3[i] -= learning_rate * gradient.b3_grad[i] / dataset->size;
        for (j = 0; j < HIDDEN_LAYER2_SIZE; j++) {
            network->W3[i][j] -= learning_rate * gradient.W3_grad[i][j] / dataset->size;
        }
    }

    return total_loss;
}
