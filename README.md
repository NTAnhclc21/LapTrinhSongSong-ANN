# LapTrinhSongSong-ANN
Đồ Án Cuối Kỳ Lập Trình Song Song HCMUS

## Architecture

## Dataset

## How to Run

### Prerequisites
Ensure you have:
- **GCC** (for CPU compilation)
- **CUDA Toolkit** (for GPU compilation)
- **Make**

### Compiling
1. **CPU Version**
   ```bash
   make main_cpu
   ```
   Generates `bin/main_cpu`.

2. **GPU Version**
   ```bash
   make main_gpu_v0
   ```
   Generates `bin/main_gpu_v0`. Use a custom architecture with:
   ```bash
   make main_gpu_v0 ARCH=<architecture>
   ```
   Default: `sm_75`.

### Running
- **CPU:**
  ```bash
  ./bin/main_cpu
  ```
- **GPU:**
  ```bash
  ./bin/main_gpu_v0
  ```

### Cleaning
Remove compiled binaries:
```bash
make clean
```

## References
- [A neural network implementation for the MNIST dataset, written in plain C](https://github.com/AndrewCarterUK/mnist-neural-network-plain-c.git), by AndrewCarterUK