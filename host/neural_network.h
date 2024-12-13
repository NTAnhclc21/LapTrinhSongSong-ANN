#ifndef NEURAL_NETWORK_H_
#define NEURAL_NETWORK_H_

#include "mnist_file.h"

#define INPUT_LAYER_SIZE MNIST_IMAGE_SIZE   // 784 nodes
#define HIDDEN_LAYER1_SIZE 128              // First hidden layer
#define HIDDEN_LAYER2_SIZE 128              // Second hidden layer
#define OUTPUT_LAYER_SIZE MNIST_LABELS      // 10 output nodes

// Activation function types
typedef enum {
    ACTIVATION_RELU,
    ACTIVATION_SIGMOID,
    ACTIVATION_TANH
} activation_type;

typedef struct {
    float W1[HIDDEN_LAYER1_SIZE][INPUT_LAYER_SIZE];
    float b1[HIDDEN_LAYER1_SIZE];

    float W2[HIDDEN_LAYER2_SIZE][HIDDEN_LAYER1_SIZE];
    float b2[HIDDEN_LAYER2_SIZE];

    float W3[OUTPUT_LAYER_SIZE][HIDDEN_LAYER2_SIZE];
    float b3[OUTPUT_LAYER_SIZE];
} neural_network_t;

typedef struct {
    float W1_grad[HIDDEN_LAYER1_SIZE][INPUT_LAYER_SIZE];
    float b1_grad[HIDDEN_LAYER1_SIZE];

    float W2_grad[HIDDEN_LAYER2_SIZE][HIDDEN_LAYER1_SIZE];
    float b2_grad[HIDDEN_LAYER2_SIZE];

    float W3_grad[OUTPUT_LAYER_SIZE][HIDDEN_LAYER2_SIZE];
    float b3_grad[OUTPUT_LAYER_SIZE];
} neural_network_gradient_t;

void neural_network_random_weights(neural_network_t * network);
float neural_network_relu(float x);
float neural_network_relu_derivative(float x);
void neural_network_hypothesis(mnist_image_t * image, neural_network_t * network, float activations[MNIST_LABELS]);
float neural_network_gradient_update(mnist_image_t * image, neural_network_t * network, neural_network_gradient_t * gradient, uint8_t label);
float neural_network_training_step(mnist_dataset_t * dataset, neural_network_t * network, float learning_rate);

#endif
