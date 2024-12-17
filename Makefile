all:
	gcc -o main main.c src/neural_network.c src/mnist_file.c src/training.c -Iinclude -lm

clean:
	rm -f main