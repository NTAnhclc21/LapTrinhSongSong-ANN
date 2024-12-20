CC = gcc
NVCC = nvcc
CFLAGS = -Iinclude
LDFLAGS = -lm

# Default architecture (can be overridden from the command line)
ARCH ?= sm_75

# Executable targets
CPU_TARGET = bin/main_cpu
GPU_TARGET_V0 = bin/main_gpu_v0
# GPU_TARGET_V1 = bin/main_gpu_v1
# GPU_TARGET_V2 = bin/main_gpu_v2

# Source files
CPU_SRCS = src/main.c src/mnist_file.c src/neural_network.c
GPU_SRCS = src/main.c src/cuda_kernels.cu src/mnist_file.c src/neural_network.c

# Build CPU version
main_cpu: $(CPU_SRCS)
    $(CC) -o $(CPU_TARGET) $(CPU_SRCS) $(CFLAGS) $(LDFLAGS)

# Build GPU version V0
main_gpu_v0: $(GPU_SRCS)
    $(NVCC) -arch=$(ARCH) -DUSE_GPU_V0 -o $(GPU_TARGET_V0) $(GPU_SRCS) $(CFLAGS) -lcudart

# # Build GPU version V1
# main_gpu_v1: $(GPU_SRCS)
# 	$(NVCC) -arch=$(ARCH) -DUSE_GPU_V1 -o $(GPU_TARGET_V1) src/main.c src/cuda_kernels_v1.cu src/mnist.c $(CFLAGS) -lcudart

# # Build GPU version V2
# main_gpu_v2: $(GPU_SRCS)
# 	$(NVCC) -arch=$(ARCH) -DUSE_GPU_V2 -o $(GPU_TARGET_V2) src/main.c src/cuda_kernels_v2.cu src/mnist.c $(CFLAGS) -lcudart

# Clean up build artifacts
clean:
    rm -f $(CPU_TARGET) $(GPU_TARGET_V0) $(GPU_TARGET_V1) $(GPU_TARGET_V2)