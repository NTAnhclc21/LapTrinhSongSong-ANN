CC = nvcc
TARGET = sweeping
SRCS = main.cu mnist_file.cu neural_network_gpu.cu

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) -o $(TARGET) $(SRCS)

clean:
	rm -f $(TARGET)


