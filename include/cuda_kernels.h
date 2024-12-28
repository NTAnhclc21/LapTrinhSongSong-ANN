#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#define CHECK(call)\
{\
    const cudaError_t error = call;\
    if (error != cudaSuccess)\
    {\
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
        fprintf(stderr, "code: %d, reason: %s\n", error,\
                cudaGetErrorString(error));\
        exit(EXIT_FAILURE);\
    }\
}

struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);
        cudaEventSynchronize(start);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    // Returns the time in seconds
    double Elapsed()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return (double)elapsed / 1000.0;
    }
};

// Matrix-vector multiplication: out = W * x + b
__global__ void matvec_mult(float *W, float *x, float *b, float *out, int rows, int cols);

// ReLU activation function: f(x) = max(0, x)
__global__ void relu_activation(float *input, int size);

// Softmax activation function: f(x[i]) = exp(x[i]) / sum(exp(x[j]))
__global__ void softmax_activation(float *input, float *output, int size);

// Compute cross-entropy loss for the output layer
__global__ void compute_output_layer_error(float *output_activations, float *output_errors, uint8_t label, int size);

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



/**
 * Functions will be defined in the following files:
 * - src/restrict/cuda_kernels.cu
 */
__global__ void optimizedKernel(float* __restrict__ d_input, float* __restrict__ d_output, 
                                 int numChannels, int width, int height);


#endif // CUDA_KERNELS_H