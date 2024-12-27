CC = nvcc
TARGET = tiled_shared_memory
SRCS = main.cu tiled_kernel.cu mnist_file.cu neural_network_gpu.cu

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) -o $(TARGET) $(SRCS)

clean:
	rm -f $(TARGET)
