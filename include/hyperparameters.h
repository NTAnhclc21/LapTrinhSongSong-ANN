#ifndef HYPERPARAMETERS_H
#define HYPERPARAMETERS_H

// MNIST-fashion file path
#define TRAIN_IMAGES    "data/train-images-idx3-ubyte"
#define TRAIN_LABELS    "data/train-labels-idx1-ubyte"
#define TEST_IMAGES     "data/t10k-images-idx3-ubyte"
#define TEST_LABELS     "data/t10k-labels-idx1-ubyte"

// Hyperparameter definitions
#define INPUT_LAYER_SIZE      784       // MNIST-fashionfashion 28x28 image
#define HIDDEN_LAYER1_SIZE    128
#define HIDDEN_LAYER2_SIZE    64
#define OUTPUT_LAYER_SIZE     10        // MNIST-fashion 10 labels

#define BATCH_SIZE            32
#define LEARNING_RATE         0.5f
#define STEPS                 1000      // GD version: Mini-batch Gradient Descent (batch/step)

#endif
