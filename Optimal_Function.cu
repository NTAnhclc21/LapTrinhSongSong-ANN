#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <float.h>

#define BLOCK_SIZE 16
#define IMAGE_SIZE 28
#define NUM_TRAIN_IMAGES 60000
#define NUM_TRAIN_LABELS 60000
#define NUM_TEST_IMAGES 10000
#define NUM_TEST_LABELS 10000
#define KERNEL_SIZE 3

using namespace cooperative_groups;

// Function to load Fashion-MNIST image data
void loadMNISTImages(const char* imagePath, float* data, int numImages) 
{
    FILE* file = fopen(imagePath, "rb");
    if (!file) 
    {
        fprintf(stderr, "Error opening file: %s\n", imagePath);
        exit(EXIT_FAILURE);
    }

    fseek(file, 16, SEEK_SET); // Skip the header
    for (int i = 0; i < numImages * IMAGE_SIZE * IMAGE_SIZE; ++i) 
    {
        unsigned char pixel;
        fread(&pixel, sizeof(unsigned char), 1, file);
        data[i] = pixel / 255.0f; // Normalize pixel value to [0, 1]
    }

    fclose(file);
}

// Function to load Fashion-MNIST label data
void loadMNISTLabels(const char* labelPath, int* labels, int numLabels) 
{
    FILE* file = fopen(labelPath, "rb");
    if (!file) 
    {
        fprintf(stderr, "Error opening file: %s\n", labelPath);
        exit(EXIT_FAILURE);
    }

    fseek(file, 8, SEEK_SET); // Skip the header
    for (int i = 0; i < numLabels; ++i) 
    {
        unsigned char label;
        fread(&label, sizeof(unsigned char), 1, file);
        labels[i] = (int)label;
    }

    fclose(file);
}

// Basic Kernel
__global__ void basicConvolutionKernel(float* input, float* kernel, float* output, int width, int height, int kernelSize) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float sum = 0.0;
    int halfKernel = kernelSize / 2;

    for (int i = -halfKernel; i <= halfKernel; i++) 
    {
        for (int j = -halfKernel; j <= halfKernel; j++) 
        {
            int nx = x + j;
            int ny = y + i;

            if (nx >= 0 && ny >= 0 && nx < width && ny < height) 
            {
                sum += input[ny * width + nx] * kernel[(i + halfKernel) * kernelSize + (j + halfKernel)];
            }
        }
    }

    output[y * width + x] = sum;
}

// Kernel with Weight Matrix in Constant Memory
__constant__ float constKernel[KERNEL_SIZE * KERNEL_SIZE];

__global__ void constantMemoryConvolutionKernel(float* input, float* output, int width, int height, int kernelSize) 
{
    __shared__ float tile[BLOCK_SIZE + 2][BLOCK_SIZE + 2];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x + 1;
    int ty = threadIdx.y + 1;

    int halfKernel = kernelSize / 2;

    tile[ty][tx] = (x < width && y < height) ? input[y * width + x] : 0.0f;

    if (threadIdx.x == 0 && x > 0) tile[ty][tx - 1] = input[y * width + x - 1];
    if (threadIdx.x == blockDim.x - 1 && x < width - 1) tile[ty][tx + 1] = input[y * width + x + 1];
    if (threadIdx.y == 0 && y > 0) tile[ty - 1][tx] = input[(y - 1) * width + x];
    if (threadIdx.y == blockDim.y - 1 && y < height - 1) tile[ty + 1][tx] = input[(y + 1) * width + x];

    __syncthreads();

    if (x >= width || y >= height) return;

    float sum = 0.0;
    for (int i = -halfKernel; i <= halfKernel; i++) 
    {
        for (int j = -halfKernel; j <= halfKernel; j++) 
        {
            sum += tile[ty + i][tx + j] * constKernel[(i + halfKernel) * kernelSize + (j + halfKernel)];
        }
    }

    output[y * width + x] = sum;
}

// Tiled Shared Memory Convolution Kernel
__global__ void tiledSharedMemoryConvolutionKernel(float* input, float* kernel, float* output, int width, int height, int kernelSize) 
{
    __shared__ float tile[BLOCK_SIZE + 2][BLOCK_SIZE + 2];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x + 1;
    int ty = threadIdx.y + 1;

    int halfKernel = kernelSize / 2;

    tile[ty][tx] = (x < width && y < height) ? input[y * width + x] : 0.0f;

    if (threadIdx.x == 0 && x > 0) tile[ty][tx - 1] = input[y * width + x - 1];
    if (threadIdx.x == blockDim.x - 1 && x < width - 1) tile[ty][tx + 1] = input[y * width + x + 1];
    if (threadIdx.y == 0 && y > 0) tile[ty - 1][tx] = input[(y - 1) * width + x];
    if (threadIdx.y == blockDim.y - 1 && y < height - 1) tile[ty + 1][tx] = input[(y + 1) * width + x];

    __syncthreads();

    if (x >= width || y >= height) return;

    float sum = 0.0;
    for (int i = -halfKernel; i <= halfKernel; i++) 
    {
        for (int j = -halfKernel; j <= halfKernel; j++) 
        {
            sum += tile[ty + i][tx + j] * kernel[(i + halfKernel) * kernelSize + (j + halfKernel)];
        }
    }

    output[y * width + x] = sum;
}

// Shared Memory Matrix Multiplication and Input Matrix Unrolling Kernel
__global__ void sharedMatrixMultiplicationKernel(float* input, float* kernel, float* output, int width, int height, int kernelSize) 
{
    __shared__ float tileA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float tileB[BLOCK_SIZE][BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * BLOCK_SIZE + ty;
    int col = blockIdx.x * BLOCK_SIZE + tx;

    float value = 0.0;

    for (int t = 0; t < (width + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) 
    {
        tileA[ty][tx] = (row < height && t * BLOCK_SIZE + tx < width) ? input[row * width + t * BLOCK_SIZE + tx] : 0.0f;
        tileB[ty][tx] = (col < width && t * BLOCK_SIZE + ty < kernelSize) ? kernel[(t * BLOCK_SIZE + ty) * kernelSize + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) 
        {
            value += tileA[ty][k] * tileB[k][tx];
        }

        __syncthreads();
    }

    if (row < height && col < width) 
    {
        output[row * width + col] = value;
    }
}

// Input Channel Reduction: Tree-based Reduction
__global__ void inputChannelReductionTreeKernel(float* input, float* output, int width, int height) 
{
    extern __shared__ float sharedData[];

    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int globalIndex = blockIdx.x * blockDim.x + threadIdx.x + blockIdx.y * blockDim.y * width;

    sharedData[tid] = (globalIndex < width * height) ? input[globalIndex] : 0.0f;
    __syncthreads();

    // Perform reduction in shared memory
    for (int stride = blockDim.x * blockDim.y / 2; stride > 0; stride >>= 1) 
    {
        if (tid < stride) 
        {
            sharedData[tid] += sharedData[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) 
    {
        output[blockIdx.x + blockIdx.y * gridDim.x] = sharedData[0];
    }
}

// Input Channel Reduction: Atomics
__global__ void inputChannelReductionAtomicsKernel(float* input, float* output, int width, int height) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) 
    {
        int index = y * width + x;
        atomicAdd(output, input[index]);
    }
}

// Fixed Point (FP16) Arithmetic Kernel
__global__ void fp16ConvolutionKernel(__half* input, __half* kernel, __half* output, int width, int height, int kernelSize) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    __half sum = __float2half(0.0f);
    int halfKernel = kernelSize / 2;

    for (int i = -halfKernel; i <= halfKernel; i++) 
    {
        for (int j = -halfKernel; j <= halfKernel; j++) 
        {
            int nx = x + j;
            int ny = y + i;

            if (nx >= 0 && ny >= 0 && nx < width && ny < height) 
            {
                sum = __hadd(sum, __hmul(input[ny * width + nx], kernel[(i + halfKernel) * kernelSize + (j + halfKernel)]));
            }
        }
    }

    output[y * width + x] = sum;
}

// Using Streams to Overlap Computation with Data Transfer
void runWithStreams(float* h_input, float* h_output, float* d_input, float* d_output, int width, int height, int kernelSize) 
{
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    size_t dataSize = width * height * sizeof(float);

    // Split the input data into two halves
    size_t halfSize = dataSize / 2;

    // Asynchronously transfer the first half of the data and process it
    cudaMemcpyAsync(d_input, h_input, halfSize, cudaMemcpyHostToDevice, stream1);
    basicConvolutionKernel<<<dim3((width / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE), dim3(BLOCK_SIZE, BLOCK_SIZE), 0, stream1>>>(d_input, d_input, d_output, width / 2, height, kernelSize);
    cudaMemcpyAsync(h_output, d_output, halfSize, cudaMemcpyDeviceToHost, stream1);

    // Asynchronously transfer the second half of the data and process it
    cudaMemcpyAsync(d_input + width * height / 2, h_input + width * height / 2, halfSize, cudaMemcpyHostToDevice, stream2);
    basicConvolutionKernel<<<dim3((width / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE), dim3(BLOCK_SIZE, BLOCK_SIZE), 0, stream2>>>(d_input + width * height / 2, d_input, d_output + width * height / 2, width / 2, height, kernelSize);
    cudaMemcpyAsync(h_output + width * height / 2, d_output + width * height / 2, halfSize, cudaMemcpyDeviceToHost, stream2);

    // Synchronize streams
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
}

// Kernel with restrict pointers and loop unrolling for optimization
__global__ void optimizedConvolutionKernel(float* __restrict__ input, float* __restrict__ kernel, float* __restrict__ output, int width, int height, int kernelSize) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float sum = 0.0f;
    int halfKernel = kernelSize / 2;

    #pragma unroll
    for (int i = -halfKernel; i <= halfKernel; i++) 
    {
        #pragma unroll
        for (int j = -halfKernel; j <= halfKernel; j++) 
        {
            int nx = x + j;
            int ny = y + i;

            if (nx >= 0 && ny >= 0 && nx < width && ny < height) 
            {
                sum += input[ny * width + nx] * kernel[(i + halfKernel) * kernelSize + (j + halfKernel)];
            }
        }
    }

    output[y * width + x] = sum;
}

// Function to sweep through different block sizes to find optimal parameters
void sweepParameters(float* d_input, float* d_kernel, float* d_output, int width, int height, int kernelSize) 
{
    int blockSizes[] = {8, 16, 32};
    float minTime = FLT_MAX;
    int bestBlockSize = 0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int b = 0; b < 3; b++) 
    {
        int blockSize = blockSizes[b];
        dim3 block(blockSize, blockSize);
        dim3 grid((width + blockSize - 1) / blockSize, (height + blockSize - 1) / blockSize);

        cudaEventRecord(start);
        optimizedConvolutionKernel<<<grid, block>>>(d_input, d_kernel, d_output, width, height, kernelSize);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float time;
        cudaEventElapsedTime(&time, start, stop);

        printf("Block size: %dx%d, Time: %f ms\n", blockSize, blockSize, time);

        if (time < minTime) 
        {
            minTime = time;
            bestBlockSize = blockSize;
        }
    }

    printf("Best block size: %dx%d with time %f ms\n", bestBlockSize, bestBlockSize, minTime);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void checkCudaErrors(cudaError_t err, const char* msg) 
{
    if (err != cudaSuccess) 
    {
        fprintf(stderr, "CUDA Error: %s - %s\n", msg, cudaGetErrorString(err));
        exit(-1);
    }
}

int main() 
{
    const int numTrainImages = NUM_TRAIN_IMAGES;
    const int imageSize = IMAGE_SIZE * IMAGE_SIZE;

    float* h_trainImages = (float*)malloc(numTrainImages * imageSize * sizeof(float));
    float* h_kernel = (float*)malloc(KERNEL_SIZE * KERNEL_SIZE * sizeof(float));
    float* h_output = (float*)malloc(numTrainImages * imageSize * sizeof(float));

    float* h_output1 = (float*)malloc(sizeof(float));
    *h_output1 = 0.0f;

    __half* h_trainImages_fp16 = (__half*)malloc(numTrainImages * imageSize * sizeof(__half));
    __half* h_output_fp16 = (__half*)malloc(numTrainImages * imageSize * sizeof(__half));
    __half* h_kernel_fp16 = (__half*)malloc(KERNEL_SIZE * KERNEL_SIZE * sizeof(__half));

    // Load Fashion-MNIST dataset
    loadMNISTImages("train-images-idx3-ubyte", h_trainImages, NUM_TRAIN_IMAGES);
    //loadMNISTLabels("train-labels-idx1-ubyte", h_trainLabels, NUM_TRAIN_LABELS);
    //loadMNISTImages("t10k-images-idx3-ubyte", h_testImages, NUM_TEST_IMAGES);
    //loadMNISTLabels("t10k-labels-idx1-ubyte", h_testLabels, NUM_TEST_LABELS);

    // Initialize kernel
    for (int i = 0; i < KERNEL_SIZE * KERNEL_SIZE; i++) 
    {
        h_kernel[i] = 0.1f;
    }

    for (int i = 0; i < numTrainImages * imageSize; ++i) 
    {
        h_trainImages_fp16[i] = __float2half(h_trainImages[i]);
    }
    for (int i = 0; i < KERNEL_SIZE * KERNEL_SIZE; ++i) 
    {
        h_kernel_fp16[i] = __float2half(0.1f);
    }

    float *d_trainImages, *d_kernel, *d_output, *d_output1;
    __half *d_trainImages_fp16, *d_output_fp16, *d_kernel_fp16;
    checkCudaErrors(cudaMalloc(&d_trainImages, numTrainImages * imageSize * sizeof(float)), "Allocating train images");
    checkCudaErrors(cudaMalloc(&d_kernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(float)), "Allocating kernel");
    checkCudaErrors(cudaMalloc(&d_output, numTrainImages * imageSize * sizeof(float)), "Allocating output");

    checkCudaErrors(cudaMalloc(&d_trainImages_fp16, numTrainImages * imageSize * sizeof(__half)), "Allocating FP16 input");
    checkCudaErrors(cudaMalloc(&d_output_fp16, numTrainImages * imageSize * sizeof(__half)), "Allocating FP16 output");
    checkCudaErrors(cudaMalloc(&d_kernel_fp16, KERNEL_SIZE * KERNEL_SIZE * sizeof(__half)), "Allocating FP16 kernel");

    checkCudaErrors(cudaMalloc(&d_output1, sizeof(float)), "Allocating output");
    checkCudaErrors(cudaMemcpy(d_trainImages, h_trainImages, numTrainImages * imageSize * sizeof(float), cudaMemcpyHostToDevice), "Copying input");
    checkCudaErrors(cudaMemcpy(d_output1, h_output1, sizeof(float), cudaMemcpyHostToDevice), "Copying initial output");

    checkCudaErrors(cudaMemcpy(d_kernel, h_kernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(float), cudaMemcpyHostToDevice), "Copying kernel");

    checkCudaErrors(cudaMemcpy(d_trainImages_fp16, h_trainImages_fp16, numTrainImages * imageSize * sizeof(__half), cudaMemcpyHostToDevice), "Copying FP16 input");
    checkCudaErrors(cudaMemcpy(d_kernel_fp16, h_kernel_fp16, KERNEL_SIZE * KERNEL_SIZE * sizeof(__half), cudaMemcpyHostToDevice), "Copying FP16 kernel");

    checkCudaErrors(cudaMemcpyToSymbol(constKernel, h_kernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(float)), "Copying kernel to constant memory");

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((IMAGE_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, (IMAGE_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEvent_t start, stop;
    float elapsedTime;

    // Measure basic kernel
    checkCudaErrors(cudaEventCreate(&start), "Creating event");
    checkCudaErrors(cudaEventCreate(&stop), "Creating event");

    checkCudaErrors(cudaEventRecord(start, 0), "Recording start event");
    basicConvolutionKernel<<<gridSize, blockSize>>>(d_trainImages, d_kernel, d_output, IMAGE_SIZE, IMAGE_SIZE, KERNEL_SIZE);
    checkCudaErrors(cudaEventRecord(stop, 0), "Recording stop event");
    checkCudaErrors(cudaEventSynchronize(stop), "Synchronizing event");
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop), "Calculating elapsed time");

    printf("Basic kernel execution time: %f ms\n", elapsedTime);

    // Measure constant memory kernel
    checkCudaErrors(cudaEventRecord(start, 0), "Recording start event");
    constantMemoryConvolutionKernel<<<gridSize, blockSize>>>(d_trainImages, d_output, IMAGE_SIZE, IMAGE_SIZE, KERNEL_SIZE);
    checkCudaErrors(cudaEventRecord(stop, 0), "Recording stop event");
    checkCudaErrors(cudaEventSynchronize(stop), "Synchronizing event");
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop), "Calculating elapsed time");

    printf("Constant memory kernel execution time: %f ms\n", elapsedTime);

    // Measure tiled shared memory kernel
    checkCudaErrors(cudaEventRecord(start, 0), "Recording start event");
    tiledSharedMemoryConvolutionKernel<<<gridSize, blockSize>>>(d_trainImages, d_kernel, d_output, IMAGE_SIZE, IMAGE_SIZE, KERNEL_SIZE);
    checkCudaErrors(cudaEventRecord(stop, 0), "Recording stop event");
    checkCudaErrors(cudaEventSynchronize(stop), "Synchronizing event");
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop), "Calculating elapsed time");

    printf("Tiled shared memory kernel execution time: %f ms\n", elapsedTime);

    // Measure shared memory matrix multiplication kernel
    checkCudaErrors(cudaEventRecord(start, 0), "Recording start event");
    sharedMatrixMultiplicationKernel<<<gridSize, blockSize>>>(d_trainImages, d_kernel, d_output, IMAGE_SIZE, IMAGE_SIZE, KERNEL_SIZE);
    checkCudaErrors(cudaEventRecord(stop, 0), "Recording stop event");
    checkCudaErrors(cudaEventSynchronize(stop), "Synchronizing event");
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop), "Calculating elapsed time");

    printf("Shared memory matrix multiplication kernel execution time: %f ms\n", elapsedTime);

    // Measure tree-based reduction kernel
    size_t sharedMemSize = blockSize.x * blockSize.y * sizeof(float);
    checkCudaErrors(cudaEventRecord(start, 0), "Recording start event");
    inputChannelReductionTreeKernel<<<gridSize, blockSize, sharedMemSize>>>(d_trainImages, d_output1, IMAGE_SIZE, IMAGE_SIZE);
    checkCudaErrors(cudaEventRecord(stop, 0), "Recording stop event");
    checkCudaErrors(cudaEventSynchronize(stop), "Synchronizing event");
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop), "Calculating elapsed time");

    printf("Tree-based reduction kernel execution time: %f ms\n", elapsedTime);

    // Measure atomic reduction kernel
    checkCudaErrors(cudaEventRecord(start, 0), "Recording start event");
    inputChannelReductionAtomicsKernel<<<gridSize, blockSize>>>(d_trainImages, d_output1, IMAGE_SIZE, IMAGE_SIZE);
    checkCudaErrors(cudaEventRecord(stop, 0), "Recording stop event");
    checkCudaErrors(cudaEventSynchronize(stop), "Synchronizing event");
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop), "Calculating elapsed time");

    printf("Atomic reduction kernel execution time: %f ms\n", elapsedTime);

    // Measure FP16 kernel
    checkCudaErrors(cudaEventRecord(start, 0), "Recording start event");
    fp16ConvolutionKernel<<<gridSize, blockSize>>>(d_trainImages_fp16, d_kernel_fp16, d_output_fp16, IMAGE_SIZE, IMAGE_SIZE, KERNEL_SIZE);
    checkCudaErrors(cudaEventRecord(stop, 0), "Recording stop event");
    checkCudaErrors(cudaEventSynchronize(stop), "Synchronizing event");
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop), "Calculating elapsed time");

    printf("FP16 kernel execution time: %f ms\n", elapsedTime);

    // Measure Using Streams kernel
    checkCudaErrors(cudaEventRecord(start, 0), "Recording start event");
    runWithStreams(h_trainImages, h_output, d_trainImages, d_output, IMAGE_SIZE, IMAGE_SIZE, KERNEL_SIZE);
    checkCudaErrors(cudaEventRecord(stop, 0), "Recording stop event");
    checkCudaErrors(cudaEventSynchronize(stop), "Synchronizing event");
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop), "Calculating elapsed time");

    printf("Using Streams kernel execution time: %f ms\n", elapsedTime);

    // Measure Tuning with restrict and Loop Unrolling kernel
    checkCudaErrors(cudaEventRecord(start, 0), "Recording start event");
    optimizedConvolutionKernel<<<gridSize, blockSize>>>(d_trainImages, d_kernel, d_output, IMAGE_SIZE, IMAGE_SIZE, KERNEL_SIZE);
    checkCudaErrors(cudaEventRecord(stop, 0), "Recording stop event");
    checkCudaErrors(cudaEventSynchronize(stop), "Synchronizing event");
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop), "Calculating elapsed time");

    printf("Tuning with restrict kernel execution time: %f ms\n", elapsedTime);

    // Measure Sweeping Various Parameters to Find Best Values kernel
    checkCudaErrors(cudaEventRecord(start, 0), "Recording start event");
    sweepParameters(d_trainImages, d_kernel, d_output, IMAGE_SIZE, IMAGE_SIZE, KERNEL_SIZE);
    checkCudaErrors(cudaEventRecord(stop, 0), "Recording stop event");
    checkCudaErrors(cudaEventSynchronize(stop), "Synchronizing event");
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop), "Calculating elapsed time");

    printf("Sweeping Various Parameters kernel execution time: %f ms\n", elapsedTime);

    checkCudaErrors(cudaMemcpy(h_output, d_output, numTrainImages * imageSize * sizeof(float), cudaMemcpyDeviceToHost), "Copying output");

    free(h_trainImages);
    free(h_kernel);
    free(h_output);
    free(h_output1);
    free(h_trainImages_fp16);
    free(h_output_fp16);
    free(h_kernel_fp16);

    cudaFree(d_trainImages);
    cudaFree(d_output);
    cudaFree(d_output1);
    cudaFree(d_trainImages_fp16);
    cudaFree(d_output_fp16);
    cudaFree(d_kernel_fp16);


    //free(h_trainLabels);
    //free(h_testImages);
    //free(h_testLabels);

    checkCudaErrors(cudaEventDestroy(start), "Destroying event");
    checkCudaErrors(cudaEventDestroy(stop), "Destroying event");

    return 0;
}
