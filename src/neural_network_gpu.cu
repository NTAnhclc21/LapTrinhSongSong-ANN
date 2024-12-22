#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "mnist_file.h"
#include "neural_network.h"
#include "hyperparameters.h"
#include "cuda_kernels.h"

// Convert a pixel value from 0-255 to one from 0 to 1
#define PIXEL_SCALE(x) (((float) (x)) / 255.0f)

// Returns a random value between 0 and 1
#define RAND_FLOAT() (((float) rand()) / ((float) RAND_MAX))

/**
 * Initialise the weights and bias vectors with values between 0 and 1
 */
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

/**
 * Use the weights and bias vector to forward propogate through the neural
 * network and calculate the activations for an input.
 */
void neural_network_hypothesis(mnist_image_t * image, neural_network_t * network, float activations[OUTPUT_LAYER_SIZE]) {
    float *d_W1, *d_W2, *d_W3, *d_b1, *d_b2, *d_b3;
    float *d_input, *d_layer1_activations, *d_layer2_activations, *d_output_activations;

    // Scale input pixels
    float scaled_pixels[INPUT_LAYER_SIZE];
    for (int i = 0; i < INPUT_LAYER_SIZE; i++) {
        scaled_pixels[i] = PIXEL_SCALE(image->pixels[i]);
    }

    // Allocate device memory
    cudaMalloc(&d_W1, HIDDEN_LAYER1_SIZE * INPUT_LAYER_SIZE * sizeof(float));
    cudaMalloc(&d_W2, HIDDEN_LAYER2_SIZE * HIDDEN_LAYER1_SIZE * sizeof(float));
    cudaMalloc(&d_W3, OUTPUT_LAYER_SIZE * HIDDEN_LAYER2_SIZE * sizeof(float));
    cudaMalloc(&d_b1, HIDDEN_LAYER1_SIZE * sizeof(float));
    cudaMalloc(&d_b2, HIDDEN_LAYER2_SIZE * sizeof(float));
    cudaMalloc(&d_b3, OUTPUT_LAYER_SIZE * sizeof(float));
    cudaMalloc(&d_input, INPUT_LAYER_SIZE * sizeof(float));
    cudaMalloc(&d_layer1_activations, HIDDEN_LAYER1_SIZE * sizeof(float));
    cudaMalloc(&d_layer2_activations, HIDDEN_LAYER2_SIZE * sizeof(float));
    cudaMalloc(&d_output_activations, OUTPUT_LAYER_SIZE * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_W1, network->W1, HIDDEN_LAYER1_SIZE * INPUT_LAYER_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, network->W2, HIDDEN_LAYER2_SIZE * HIDDEN_LAYER1_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W3, network->W3, OUTPUT_LAYER_SIZE * HIDDEN_LAYER2_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1, network->b1, HIDDEN_LAYER1_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2, network->b2, HIDDEN_LAYER2_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b3, network->b3, OUTPUT_LAYER_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, scaled_pixels, INPUT_LAYER_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Launch CUDA kernels
    dim3 blockDim(128);
    dim3 gridDim1((HIDDEN_LAYER1_SIZE + blockDim.x - 1) / blockDim.x);
    dim3 gridDim2((HIDDEN_LAYER2_SIZE + blockDim.x - 1) / blockDim.x);
    dim3 gridDim3((OUTPUT_LAYER_SIZE + blockDim.x - 1) / blockDim.x);

    matvec_mult<<<gridDim1, blockDim>>>(d_W1, d_input, d_b1, d_layer1_activations, HIDDEN_LAYER1_SIZE, INPUT_LAYER_SIZE);
    relu_activation<<<gridDim1, blockDim>>>(d_layer1_activations, HIDDEN_LAYER1_SIZE);

    matvec_mult<<<gridDim2, blockDim>>>(d_W2, d_layer1_activations, d_b2, d_layer2_activations, HIDDEN_LAYER2_SIZE, HIDDEN_LAYER1_SIZE);
    relu_activation<<<gridDim2, blockDim>>>(d_layer2_activations, HIDDEN_LAYER2_SIZE);

    matvec_mult<<<gridDim3, blockDim>>>(d_W3, d_layer2_activations, d_b3, d_output_activations, OUTPUT_LAYER_SIZE, HIDDEN_LAYER2_SIZE);
    softmax_activation<<<1, blockDim>>>(d_output_activations, d_output_activations, OUTPUT_LAYER_SIZE);

    // Copy results back to host
    cudaMemcpy(activations, d_output_activations, OUTPUT_LAYER_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_W1);
    cudaFree(d_W2);
    cudaFree(d_W3);
    cudaFree(d_b1);
    cudaFree(d_b2);
    cudaFree(d_b3);
    cudaFree(d_input);
    cudaFree(d_layer1_activations);
    cudaFree(d_layer2_activations);
    cudaFree(d_output_activations);
}

/**
 * Update the gradients for this step of gradient descent using the gradient
 * contributions from a single training example (image).
 * 
 * This function returns the loss contribution from this training example.
 */
float neural_network_gradient_update(mnist_image_t * image, neural_network_t * network, neural_network_gradient_t * gradient, uint8_t label) {
    float *d_W1, *d_W2, *d_W3, *d_b1, *d_b2, *d_b3;
    float *d_input, *d_layer1_activations, *d_layer2_activations, *d_output_activations;
    float *d_layer2_errors, *d_layer1_errors, *d_output_errors;
    float *d_W1_grad, *d_W2_grad, *d_W3_grad, *d_b1_grad, *d_b2_grad, *d_b3_grad;

    // Scale input pixels
    float scaled_pixels[INPUT_LAYER_SIZE];
    for (int i = 0; i < INPUT_LAYER_SIZE; i++) {
        scaled_pixels[i] = PIXEL_SCALE(image->pixels[i]);
    }

    // Allocate device memory
    cudaMalloc(&d_W1, HIDDEN_LAYER1_SIZE * INPUT_LAYER_SIZE * sizeof(float));
    cudaMalloc(&d_W2, HIDDEN_LAYER2_SIZE * HIDDEN_LAYER1_SIZE * sizeof(float));
    cudaMalloc(&d_W3, OUTPUT_LAYER_SIZE * HIDDEN_LAYER2_SIZE * sizeof(float));
    cudaMalloc(&d_b1, HIDDEN_LAYER1_SIZE * sizeof(float));
    cudaMalloc(&d_b2, HIDDEN_LAYER2_SIZE * sizeof(float));
    cudaMalloc(&d_b3, OUTPUT_LAYER_SIZE * sizeof(float));
    cudaMalloc(&d_input, INPUT_LAYER_SIZE * sizeof(float));
    cudaMalloc(&d_layer1_activations, HIDDEN_LAYER1_SIZE * sizeof(float));
    cudaMalloc(&d_layer2_activations, HIDDEN_LAYER2_SIZE * sizeof(float));
    cudaMalloc(&d_output_activations, OUTPUT_LAYER_SIZE * sizeof(float));
    cudaMalloc(&d_layer2_errors, HIDDEN_LAYER2_SIZE * sizeof(float));
    cudaMalloc(&d_layer1_errors, HIDDEN_LAYER1_SIZE * sizeof(float));
    cudaMalloc(&d_output_errors, OUTPUT_LAYER_SIZE * sizeof(float));
    cudaMalloc(&d_W1_grad, HIDDEN_LAYER1_SIZE * INPUT_LAYER_SIZE * sizeof(float));
    cudaMalloc(&d_W2_grad, HIDDEN_LAYER2_SIZE * HIDDEN_LAYER1_SIZE * sizeof(float));
    cudaMalloc(&d_W3_grad, OUTPUT_LAYER_SIZE * HIDDEN_LAYER2_SIZE * sizeof(float));
    cudaMalloc(&d_b1_grad, HIDDEN_LAYER1_SIZE * sizeof(float));
    cudaMalloc(&d_b2_grad, HIDDEN_LAYER2_SIZE * sizeof(float));
    cudaMalloc(&d_b3_grad, OUTPUT_LAYER_SIZE * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_W1, network->W1, HIDDEN_LAYER1_SIZE * INPUT_LAYER_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, network->W2, HIDDEN_LAYER2_SIZE * HIDDEN_LAYER1_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W3, network->W3, OUTPUT_LAYER_SIZE * HIDDEN_LAYER2_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1, network->b1, HIDDEN_LAYER1_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2, network->b2, HIDDEN_LAYER2_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b3, network->b3, OUTPUT_LAYER_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, scaled_pixels, INPUT_LAYER_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Launch CUDA kernels for forward pass
    dim3 blockDim(128);
    dim3 gridDim1((HIDDEN_LAYER1_SIZE + blockDim.x - 1) / blockDim.x);
    dim3 gridDim2((HIDDEN_LAYER2_SIZE + blockDim.x - 1) / blockDim.x);
    dim3 gridDim3((OUTPUT_LAYER_SIZE + blockDim.x - 1) / blockDim.x);

    matvec_mult<<<gridDim1, blockDim>>>(d_W1, d_input, d_b1, d_layer1_activations, HIDDEN_LAYER1_SIZE, INPUT_LAYER_SIZE);
    cudaDeviceSynchronize();
    relu_activation<<<gridDim1, blockDim>>>(d_layer1_activations, HIDDEN_LAYER1_SIZE);
    cudaDeviceSynchronize();

    matvec_mult<<<gridDim2, blockDim>>>(d_W2, d_layer1_activations, d_b2, d_layer2_activations, HIDDEN_LAYER2_SIZE, HIDDEN_LAYER1_SIZE);
    cudaDeviceSynchronize();
    relu_activation<<<gridDim2, blockDim>>>(d_layer2_activations, HIDDEN_LAYER2_SIZE);
    cudaDeviceSynchronize();

    matvec_mult<<<gridDim3, blockDim>>>(d_W3, d_layer2_activations, d_b3, d_output_activations, OUTPUT_LAYER_SIZE, HIDDEN_LAYER2_SIZE);
    cudaDeviceSynchronize();
    softmax_activation<<<1, blockDim>>>(d_output_activations, d_output_activations, OUTPUT_LAYER_SIZE);
    cudaDeviceSynchronize();

    // Copy output activations back to host
    cudaMemcpy(output_activations, d_output_activations, OUTPUT_LAYER_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // Backpropagation
    // Output layer error
    for (int i = 0; i < OUTPUT_LAYER_SIZE; i++) {
        output_errors[i] = (i == label) ? output_activations[i] - 1 : output_activations[i];
    }

    // Copy output errors to device
    cudaMemcpy(d_output_errors, output_errors, OUTPUT_LAYER_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Backpropagation
    // Output layer error
    compute_output_layer_error<<<gridDim3, blockDim>>>(d_output_activations, d_output_errors, label, OUTPUT_LAYER_SIZE);
    cudaDeviceSynchronize();

    // Hidden layer 2 error
    compute_hidden_layer_error<<<gridDim2, blockDim>>>(d_output_errors, d_W3, d_layer2_activations, d_layer2_errors, OUTPUT_LAYER_SIZE, HIDDEN_LAYER2_SIZE);
    cudaDeviceSynchronize();

    // Hidden layer 1 error
    compute_hidden_layer_error<<<gridDim1, blockDim>>>(d_layer2_errors, d_W2, d_layer1_activations, d_layer1_errors, HIDDEN_LAYER2_SIZE, HIDDEN_LAYER1_SIZE);
    cudaDeviceSynchronize();

    // Accumulate gradients for output layer
    accumulate_gradients<<<gridDim3, blockDim>>>(d_W3, d_b3, d_output_errors, d_layer2_activations, d_W3_grad, d_b3_grad, OUTPUT_LAYER_SIZE, HIDDEN_LAYER2_SIZE);
    cudaDeviceSynchronize();

    // Accumulate gradients for hidden layer 2
    accumulate_gradients<<<gridDim2, blockDim>>>(d_W2, d_b2, d_layer2_errors, d_layer1_activations, d_W2_grad, d_b2_grad, HIDDEN_LAYER2_SIZE, HIDDEN_LAYER1_SIZE);
    cudaDeviceSynchronize();

    // Accumulate gradients for hidden layer 1
    accumulate_gradients<<<gridDim1, blockDim>>>(d_W1, d_b1, d_layer1_errors, d_input, d_W1_grad, d_b1_grad, HIDDEN_LAYER1_SIZE, INPUT_LAYER_SIZE);
    cudaDeviceSynchronize();

    // Copy gradients back to host
    cudaMemcpy(gradient->W1_grad, d_W1_grad, HIDDEN_LAYER1_SIZE * INPUT_LAYER_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(gradient->W2_grad, d_W2_grad, HIDDEN_LAYER2_SIZE * HIDDEN_LAYER1_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(gradient->W3_grad, d_W3_grad, OUTPUT_LAYER_SIZE * HIDDEN_LAYER2_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(gradient->b1_grad, d_b1_grad, HIDDEN_LAYER1_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(gradient->b2_grad, d_b2_grad, HIDDEN_LAYER2_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(gradient->b3_grad, d_b3_grad, OUTPUT_LAYER_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_W1);
    cudaFree(d_W2);
    cudaFree(d_W3);
    cudaFree(d_b1);
    cudaFree(d_b2);
    cudaFree(d_b3);
    cudaFree(d_input);
    cudaFree(d_layer1_activations);
    cudaFree(d_layer2_activations);
    cudaFree(d_output_activations);
    cudaFree(d_layer2_errors);
    cudaFree(d_layer1_errors);
    cudaFree(d_output_errors);
    cudaFree(d_W1_grad);
    cudaFree(d_W2_grad);
    cudaFree(d_W3_grad);
    cudaFree(d_b1_grad);
    cudaFree(d_b2_grad);
    cudaFree(d_b3_grad);

    // Cross-entropy loss for the output
    return 0.0f - log(output_activations[label]);
}

/**
 * Run one step of gradient descent and update the neural network. Return the total loss (sum of loss)
 */
float neural_network_training_step(mnist_dataset_t * dataset, neural_network_t * network, float learning_rate) {
    float total_loss = 0.0f;
    int i, j;

    // Initialize gradients to zero
    neural_network_gradient_t gradient;
    memset(&gradient, 0, sizeof(neural_network_gradient_t));

    /**
     * Calculate the Gradients and the Cross-Entropy Loss by looping through the training set.
     * The returned gradient is the sum of gradients from all inputs, not the average.
     */
    for (i = 0; i < dataset->size; i++) {
        total_loss += neural_network_gradient_update_gpu_v0(&dataset->images[i], network, &gradient, dataset->labels[i]);
    }

    // Update weights and biases
    for (i = 0; i < HIDDEN_LAYER1_SIZE; i++) {
        network->b1[i] -= learning_rate * gradient.b1_grad[i] / batch->size;
        for (j = 0; j < INPUT_LAYER_SIZE; j++) {
            network->W1[i][j] -= learning_rate * gradient.W1_grad[i][j] / batch->size;
        }
    }

    for (i = 0; i < HIDDEN_LAYER2_SIZE; i++) {
        network->b2[i] -= learning_rate * gradient.b2_grad[i] / batch->size;
        for (j = 0; j < HIDDEN_LAYER1_SIZE; j++) {
            network->W2[i][j] -= learning_rate * gradient.W2_grad[i][j] / batch->size;
        }
    }

    for (i = 0; i < OUTPUT_LAYER_SIZE; i++) {
        network->b3[i] -= learning_rate * gradient.b3_grad[i] / batch->size;
        for (j = 0; j < HIDDEN_LAYER2_SIZE; j++) {
            network->W3[i][j] -= learning_rate * gradient.W3_grad[i][j] / batch->size;
        }
    }
    
    return total_loss;
}

/**
 * Calculate the accuracy of the predictions of a neural network on a dataset.
 */
float calculate_accuracy(mnist_dataset_t * dataset, neural_network_t * network)
{
    float activations[MNIST_LABELS], max_activation;
    int i, j, correct, predict;

    // Loop through the dataset
    for (i = 0, correct = 0; i < dataset->size; i++) {
        // Calculate the activations for each image using the neural network
        neural_network_hypothesis(&dataset->images[i], network, activations);

        // Set predict to the index of the greatest activation
        for (j = 0, predict = 0, max_activation = activations[0]; j < MNIST_LABELS; j++) {
            if (max_activation < activations[j]) {
                max_activation = activations[j];
                predict = j;
            }
        }

        // Increment the correct count if we predicted the right label
        if (predict == dataset->labels[i]) {
            correct++;
        }
    }

    // Return the percentage we predicted correctly as the accuracy
    return ((float) correct) / ((float) dataset->size);
}

