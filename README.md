# PFPL
This repository holds the latest versions of the PFPL code that was published in IPDPS 2025.
PFPL is a guaranteed-error bound lossy compressor/decompressor that produces bit-for-bit identical files on CPUs and GPUs. It supports absolute (ABS), relative (REL), and normalized-absolute (NOA) error bounds. It supports both single- and double-precision inputs in IEEE 754 binary format.

### Check-list (meta-information)
  - Algorithm: Single- and double-precision lossy ABS, REL, and NOA compressors and decompressors
  - Compilation: g++ and nvcc
  - Data set: SDRBench (or IEEE 754 floating-point input of your choice)
  - Hardware: CPU and (optionally) GPU 
  - Execution: Parallel
  - Output: Compressed and lossily reconstructed files
  - Code license: BSD 3-Clause License
  - Corresponding paper: TODO insert DOI

### Description

###### Hardware dependencies

The hardware required for these codes are an x86 multi-core CPU and a CUDA-capable GPU. We used a 16-core AMD Threadripper 2950X CPU @ 3.5 GHz with hyperthreading enabled to run the CPU codes. To run the GPU codes, we used an NVIDIA RTX 4090. Using similar hardware should result in throughputs similar to those reported in the paper.

###### Software dependencies

The required software includes:
- The computational artifact from https://github.com/burtscher/PFPL
- GCC 7.5.0 or higher
- OpenMP 3.1 or higher
- CUDA 11.0 or higher
- Make

###### Data sets

The data sets used in the paper can be found at https://sdrbench.github.io.

###### Installation

To install the code, perform the following steps:
- Clone the repository from https://github.com/burtscher/PFPL
- Run 'make all'
- The executables can be compiled separately as well
- The compiled executables will be in bin/[f32 or f64]/[serial, openmp, or gpu]/

> **Note:**  
> By default, the makefile builds for CUDA -arch=sm_70. Set this parameter to match the target GPU's compute capability before compiling.

### Running the codes

After compiling, the codes can be run as outlined in the following examples:
    ./f32_abs_compress_cuda input_file_name compressed_file_name error_bound [threshold]
    ./f32_abs_decompress_cuda compressed_file_name reconstructed_file_name

The threshold parameter will trigger any absolute value at or above the threshold to be losslessly preserved. This is to accommodate any protected values (such as sentinel values).
To match the paper, the codes will be run 9 times and the runtimes reported in seconds.
