#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#include <cuda_runtime.h>

// Matrix-vector multiplication: out = W * x + b
__global__ void matvec_mult(float *W, float *x, float *b, float *out, int rows, int cols);

// ReLU activation function: f(x) = max(0, x)
__global__ void relu_activation(float *input, int size);

// Softmax activation function: f(x[i]) = exp(x[i]) / sum(exp(x[j]))
__global__ void softmax_activation(float *input, float *output, int size);

// Compute error for hidden layers during backpropagation
__global__ void compute_hidden_layer_error(float *next_layer_errors, 
                                         float *next_layer_weights, 
                                         float *layer_activations, 
                                         float *layer_errors, 
                                         int next_layer_size, 
                                         int layer_size);

// Accumulate gradients for weights and biases
__global__ void accumulate_gradients(float *weights, 
                                   float *biases, 
                                   float *layer_errors, 
                                   float *prev_layer_activations, 
                                   float *weights_grad, 
                                   float *biases_grad, 
                                   int layer_size, 
                                   int prev_layer_size);

#endif // CUDA_KERNELS_H