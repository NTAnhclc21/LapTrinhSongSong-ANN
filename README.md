# FITUS - Parallel Programming 21KHMT - Final Project
This project involves implementing a simple Artificial Neural Network (ANN) focusing on CUDA's capabilities for optimization in parallel computing.

## Implementation

The ANN is developed using plain C for the sequential version and CUDA C for the parallel versions. The model's architecture and hyperparameters are configurable via the `hyperparameters.h` file in the `include` folder. The implementation is evaluated using the Fashion-MNIST dataset.

The implemented versions:
- `sequential` : 
  - Version name: 
- `basic_parallel` :
- `tiled` : 
- `tiled_fp16` :
- `restrict` :

## How to Run Makefile

After cloning the repository, change the directory to the project folder, in which there is a `Makefile`. Use the `make` command to compile the exectable file inside the `bin` folder.

```bash
make <version_name_>
```

The code generates a `bin/<version_name>` exectable file. Replace `version_name` with the name of the implemented version in the [Implementation](#implementation) section. For parallel versions, use a custom architecture with:

```bash
make <parallel_version_name> ARCH=<architecture>
```

The general architecture is `sm_<major><minor>` . The default architecture is `sm_75`  that specify the GPU compute capability `7.5` .

Finally, you can remove your compiled binaries with:

```bash
make clean
```

## References
- [A neural network implementation for the MNIST dataset, written in plain C](https://github.com/AndrewCarterUK/mnist-neural-network-plain-c.git), by AndrewCarterUK