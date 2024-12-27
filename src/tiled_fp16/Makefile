# Compiler settings
CC = nvcc
TARGET = fp16
SRCS = main.cu mnist_file.cu neural_network_gpu.cu  # You can add more source files here

# CUDA architecture and flags
CUDA_FLAGS = -arch=sm_60  # Adjust to your target architecture (sm_60 is for Pascal, adjust accordingly)

# Default target
all: $(TARGET)

# Compile and link directly into the executable
$(TARGET): $(SRCS)
	$(CC) $(CUDA_FLAGS) -o $(TARGET) $(SRCS)

# Clean up build files
clean:
	rm -f $(TARGET)



