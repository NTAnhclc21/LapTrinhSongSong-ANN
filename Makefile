CC = gcc
NVCC = nvcc
CFLAGS = -Iinclude
LDFLAGS = -lm

# Default architecture (can be overridden from the command line)
ARCH ?= sm_75

# Executable targets
CPU_TARGET = bin/main_cpu
GPU_TARGET_V0 = bin/main_gpu_v0
GPU_TARGET_V1 = bin/main_gpu_v1
GPU_TARGET_V2 = bin/main_gpu_v2
GPU_TARGET_V3 = bin/main_gpu_v3

# Source files
CPU_SRCS = src/sequential/main.c src/sequential/mnist_file.c src/sequential/neural_network_cpu.c
GPU_V0_SRCS = src/basic_parallel/main.cu src/basic_parallel/cuda_kernels.cu src/basic_parallel/mnist_file.cu src/basic_parallel/neural_network_gpu.cu
GPU_V1_SRCS = src/tiled/main.cu src/tiled/cuda_kernels.cu src/tiled/mnist_file.cu src/tiled/neural_network_gpu.cu
GPU_V2_SRCS = src/tiled_fp16/main.cu src/tiled_fp16/cuda_kernels.cu src/tiled_fp16/mnist_file.cu src/tiled_fp16/neural_network_gpu.cu
GPU_V3_SRCS = src/restrict/main.cu src/restrict/cuda_kernels.cu src/restrict/mnist_file.cu src/restrict/neural_network_gpu.cu

# Build CPU version
main_cpu: $(CPU_SRCS)
	$(CC) -o $(CPU_TARGET) $(CPU_SRCS) $(CFLAGS) $(LDFLAGS)

# Build GPU version V0
main_gpu_v0: $(GPU_V0_SRCS)
	$(NVCC) -arch=$(ARCH) -o $(GPU_TARGET_V0) $(GPU_V0_SRCS) $(CFLAGS) -lcudart
	$(NVCC) -arch=$(ARCH) -o $(GPU_TARGET_V1) $(GPU_V1_SRCS) $(CFLAGS) -lcudart

# Build GPU version V1
main_gpu_v1: $(GPU_V1_SRCS)
	$(NVCC) -arch=$(ARCH) -o $(GPU_TARGET_V1) $(GPU_V1_SRCS) $(CFLAGS) -lcudart

# Build GPU version V2
main_gpu_v2: $(GPU_V2_SRCS)
	$(NVCC) -arch=$(ARCH) -o $(GPU_TARGET_V2) $(GPU_V2_SRCS) $(CFLAGS) -lcudart

# Build GPU version V3
main_gpu_v3: $(GPU_V3_SRCS)
	$(NVCC) -arch=$(ARCH) -o $(GPU_TARGET_V3) $(GPU_V3_SRCS) $(CFLAGS) -lcudart

# Clean up build artifacts
clean:
	m -f $(CPU_TARGET) $(GPU_TARGET_V0) $(GPU_TARGET_V1) $(GPU_TARGET_V2) $(GPU_TARGET_V3)