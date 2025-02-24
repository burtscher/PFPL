NV_SM := 70

all: serial openmp gpu

serial:
	mkdir -p bin/f32/serial/
	mkdir -p bin/f64/serial/
	g++ -O3 -march=native -mno-fma -ffp-contract=off -I./src/ -std=c++17 -o bin/f32/serial/abs_compress_ser src/f32_abs_comp_cpu.cpp
	g++ -O3 -march=native -mno-fma -ffp-contract=off -I./src/ -std=c++17 -o bin/f32/serial/abs_decompress_ser src/f32_abs_decomp_cpu.cpp
	g++ -O3 -march=native -mno-fma -ffp-contract=off -I./src/ -std=c++17 -o bin/f32/serial/rel_compress_ser src/f32_rel_comp_cpu.cpp
	g++ -O3 -march=native -mno-fma -ffp-contract=off -I./src/ -std=c++17 -o bin/f32/serial/rel_decompress_ser src/f32_rel_decomp_cpu.cpp
	g++ -O3 -march=native -mno-fma -ffp-contract=off -I./src/ -std=c++17 -o bin/f32/serial/noa_compress_ser src/f32_noa_comp_cpu.cpp
	g++ -O3 -march=native -mno-fma -ffp-contract=off -I./src/ -std=c++17 -o bin/f32/serial/noa_decompress_ser src/f32_noa_decomp_cpu.cpp
	g++ -O3 -march=native -mno-fma -ffp-contract=off -I./src/ -std=c++17 -o bin/f64/serial/abs_compress_ser src/f64_abs_comp_cpu.cpp
	g++ -O3 -march=native -mno-fma -ffp-contract=off -I./src/ -std=c++17 -o bin/f64/serial/abs_decompress_ser src/f64_abs_decomp_cpu.cpp
	g++ -O3 -march=native -mno-fma -ffp-contract=off -I./src/ -std=c++17 -o bin/f64/serial/rel_compress_ser src/f64_rel_comp_cpu.cpp
	g++ -O3 -march=native -mno-fma -ffp-contract=off -I./src/ -std=c++17 -o bin/f64/serial/rel_decompress_ser src/f64_rel_decomp_cpu.cpp
	g++ -O3 -march=native -mno-fma -ffp-contract=off -I./src/ -std=c++17 -o bin/f64/serial/noa_compress_ser src/f64_noa_comp_cpu.cpp
	g++ -O3 -march=native -mno-fma -ffp-contract=off -I./src/ -std=c++17 -o bin/f64/serial/noa_decompress_ser src/f64_noa_decomp_cpu.cpp

openmp:
	mkdir -p bin/f32/openmp/
	mkdir -p bin/f64/openmp/
	g++ -O3 -march=native -fopenmp -mno-fma -ffp-contract=off -I./src/ -std=c++17 -o bin/f32/openmp/abs_compress_omp src/f32_abs_comp_cpu.cpp
	g++ -O3 -march=native -fopenmp -mno-fma -ffp-contract=off -I./src/ -std=c++17 -o bin/f32/openmp/abs_decompress_omp src/f32_abs_decomp_cpu.cpp
	g++ -O3 -march=native -fopenmp -mno-fma -ffp-contract=off -I./src/ -std=c++17 -o bin/f32/openmp/rel_compress_omp src/f32_rel_comp_cpu.cpp
	g++ -O3 -march=native -fopenmp -mno-fma -ffp-contract=off -I./src/ -std=c++17 -o bin/f32/openmp/rel_decompress_omp src/f32_rel_decomp_cpu.cpp
	g++ -O3 -march=native -fopenmp -mno-fma -ffp-contract=off -I./src/ -std=c++17 -o bin/f32/openmp/noa_compress_omp src/f32_noa_comp_cpu.cpp
	g++ -O3 -march=native -fopenmp -mno-fma -ffp-contract=off -I./src/ -std=c++17 -o bin/f32/openmp/noa_decompress_omp src/f32_noa_decomp_cpu.cpp
	g++ -O3 -march=native -fopenmp -mno-fma -ffp-contract=off -I./src/ -std=c++17 -o bin/f64/openmp/abs_compress_omp src/f64_abs_comp_cpu.cpp
	g++ -O3 -march=native -fopenmp -mno-fma -ffp-contract=off -I./src/ -std=c++17 -o bin/f64/openmp/abs_decompress_omp src/f64_abs_decomp_cpu.cpp
	g++ -O3 -march=native -fopenmp -mno-fma -ffp-contract=off -I./src/ -std=c++17 -o bin/f64/openmp/rel_compress_omp src/f64_rel_comp_cpu.cpp
	g++ -O3 -march=native -fopenmp -mno-fma -ffp-contract=off -I./src/ -std=c++17 -o bin/f64/openmp/rel_decompress_omp src/f64_rel_decomp_cpu.cpp
	g++ -O3 -march=native -fopenmp -mno-fma -ffp-contract=off -I./src/ -std=c++17 -o bin/f64/openmp/noa_compress_omp src/f64_noa_comp_cpu.cpp
	g++ -O3 -march=native -fopenmp -mno-fma -ffp-contract=off -I./src/ -std=c++17 -o bin/f64/openmp/noa_decompress_omp src/f64_noa_decomp_cpu.cpp

gpu:
	mkdir -p bin/f32/gpu/
	mkdir -p bin/f64/gpu/
	nvcc -O3 -arch=sm_$(NV_SM) -fmad=false -Xcompiler "-O3 -march=native -fopenmp -mno-fma -ffp-contract=off" -I./src/ -o bin/f32/gpu/f32_abs_compress_cuda src/f32_abs_comp_gpu.cu
	nvcc -O3 -arch=sm_$(NV_SM) -fmad=false -Xcompiler "-O3 -march=native -fopenmp -mno-fma -ffp-contract=off" -I./src/ -o bin/f32/gpu/f32_abs_decompress_cuda src/f32_abs_decomp_gpu.cu
	nvcc -O3 -arch=sm_$(NV_SM) -fmad=false -Xcompiler "-O3 -march=native -fopenmp -mno-fma -ffp-contract=off" -I./src/ -o bin/f32/gpu/f32_rel_compress_cuda src/f32_rel_comp_gpu.cu
	nvcc -O3 -arch=sm_$(NV_SM) -fmad=false -Xcompiler "-O3 -march=native -fopenmp -mno-fma -ffp-contract=off" -I./src/ -o bin/f32/gpu/f32_rel_decompress_cuda src/f32_rel_decomp_gpu.cu
	nvcc -O3 -arch=sm_$(NV_SM) -fmad=false -Xcompiler "-O3 -march=native -fopenmp -mno-fma -ffp-contract=off" -I./src/ -o bin/f32/gpu/f32_noa_compress_cuda src/f32_noa_comp_gpu.cu
	nvcc -O3 -arch=sm_$(NV_SM) -fmad=false -Xcompiler "-O3 -march=native -fopenmp -mno-fma -ffp-contract=off" -I./src/ -o bin/f32/gpu/f32_noa_decompress_cuda src/f32_noa_decomp_gpu.cu
	nvcc -O3 -arch=sm_$(NV_SM) -fmad=false -Xcompiler "-O3 -march=native -fopenmp -mno-fma -ffp-contract=off" -I./src/ -o bin/f64/gpu/f64_abs_compress_cuda src/f64_abs_comp_gpu.cu
	nvcc -O3 -arch=sm_$(NV_SM) -fmad=false -Xcompiler "-O3 -march=native -fopenmp -mno-fma -ffp-contract=off" -I./src/ -o bin/f64/gpu/f64_abs_decompress_cuda src/f64_abs_decomp_gpu.cu
	nvcc -O3 -arch=sm_$(NV_SM) -fmad=false -Xcompiler "-O3 -march=native -fopenmp -mno-fma -ffp-contract=off" -I./src/ -o bin/f64/gpu/f64_rel_compress_cuda src/f64_rel_comp_gpu.cu
	nvcc -O3 -arch=sm_$(NV_SM) -fmad=false -Xcompiler "-O3 -march=native -fopenmp -mno-fma -ffp-contract=off" -I./src/ -o bin/f64/gpu/f64_rel_decompress_cuda src/f64_rel_decomp_gpu.cu
	nvcc -O3 -arch=sm_$(NV_SM) -fmad=false -Xcompiler "-O3 -march=native -fopenmp -mno-fma -ffp-contract=off" -I./src/ -o bin/f64/gpu/f64_noa_compress_cuda src/f64_noa_comp_gpu.cu
	nvcc -O3 -arch=sm_$(NV_SM) -fmad=false -Xcompiler "-O3 -march=native -fopenmp -mno-fma -ffp-contract=off" -I./src/ -o bin/f64/gpu/f64_noa_decompress_cuda src/f64_noa_decomp_gpu.cu

clean:
	rm -rf bin/ 