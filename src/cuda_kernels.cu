/**
 *  Implements CUDA kernels for:
 *      Feedforward Computations:
 *          Matrix-vector multiplication.
 *          ReLU activation.
 *          Softmax activation.
 *      Backpropagation Computations:
 *          Error Calculation for Output Layer
 *          Gradient Computations (Matrix multiplications and reductions (summing gradients)
 *          Backpropagating the Error (Transpose and multiply operation)
 *          Weight and Bias Updates 
 */

#include <math.h>
#include <stdint.h>
#include <float.h>

// CUDA Kernel: Matrix-Vector Multiplication
__global__ void matvec_mult(float *W, float *x, float *b, float *out, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        float sum = b[row];
        for (int j = 0; j < cols; j++) {
            sum += W[row * cols + j] * x[j];
        }
        out[row] = sum;
    }
}

// CUDA Kernel: ReLU Activation
__global__ void relu_activation(float *input, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) input[i] = fmaxf(0.0f, input[i]);
}

// CUDA Kernel: Softmax Activation
__global__ void softmax_activation(float *input, float *output, int size) {
    __shared__ float shared_max[128];
    __shared__ float shared_sum[128];
    
    int tid = threadIdx.x;
    
    // Find max for numerical stability
    float local_max = -FLT_MAX;
    for (int i = tid; i < size; i += blockDim.x) {
        local_max = fmaxf(local_max, input[i]);
    }
    shared_max[tid] = local_max;
    __syncthreads();
    
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + stride]);
        }
        __syncthreads();
    }
    float max_val = shared_max[0];
    __syncthreads();

    // Compute exp(x - max) for stability
    float local_sum = 0.0f;
    for (int i = tid; i < size; i += blockDim.x) {
        float val = expf(input[i] - max_val);
        output[i] = val;  // Store intermediate result
        local_sum += val;
    }
    shared_sum[tid] = local_sum;
    __syncthreads();
    
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }
    float sum = shared_sum[0];
    
    // Normalize
    for (int i = tid; i < size; i += blockDim.x) {
        output[i] /= sum;
    }
}

__global__ void compute_output_layer_error(float *output_activations, float *output_errors, uint8_t label, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        output_errors[i] = (i == label) ? output_activations[i] - 1 : output_activations[i];
    }
}

__global__ void compute_hidden_layer_error(float *next_layer_errors, float *next_layer_weights, float *layer_activations, float *layer_errors, int next_layer_size, int layer_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < layer_size) {
        float error = 0.0f;
        for (int j = 0; j < next_layer_size; j++) {
            error += next_layer_errors[j] * next_layer_weights[j * layer_size + i];
        }
        layer_errors[i] = error * (layer_activations[i] > 0 ? 1.0f : 0.0f); // ReLU derivative
    }
}

__global__ void accumulate_gradients(float *weights, float *biases, float *errors, float *prev_activations, float *weights_grad, float *biases_grad, int layer_size, int prev_layer_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < layer_size) {
        atomicAdd(&biases_grad[i], errors[i]);
        for (int j = 0; j < prev_layer_size; j++) {
            atomicAdd(&weights_grad[i * prev_layer_size + j], errors[i] * prev_activations[j]);
        }
    }
}