/*
This file is part of the LC framework for synthesizing high-speed parallel lossless and error-bounded lossy data compression and decompression algorithms for CPUs and GPUs.

BSD 3-Clause License

Copyright (c) 2021-2025, Noushin Azami, Alex Fallin, Brandon Burtchell, Andrew Rodriguez, Benila Jerald, Yiqian Liu, and Martin Burtscher
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

URL: The latest version of this code is available at https://github.com/burtscher/LC-framework.

Sponsor: This code is based upon work supported by the U.S. Department of Energy, Office of Science, Office of Advanced Scientific Research (ASCR), under contract DE-SC0022223.
*/


#define NDEBUG

using byte = unsigned char;
static const int CS = 1024 * 16;  // chunk size (in bytes) [must be multiple of 8]
static const int TPB = 512;  // threads per block [must be power of 2 and at least 128]
#define WS 32


#include <cmath>
#include <string>
#include <cassert>
#include <stdexcept>
#include <cuda.h>
#include "include/macros.h"
#include "include/sum_reduction.h"
#include "include/max_scan.h"
#include "include/prefix_sum.h"
#include "components/d_DIFFNB_4.h"
#include "components/d_BIT_4.h"
#include "components/d_RZE_1.h"
#include <sys/time.h>


const int NUM_RUNS = 9;


static __device__ inline float d_REL_log2approxf(const float orig_f)
{
  //assert(orig_f > 0);
  const int mantissabits = 23;
  const int orig_i = *((int*)&orig_f);
  const int expo = (orig_i >> mantissabits) & 0xff;
  //if ((expo == 0) || (expo == 0xff)) return orig_f;
  const int frac_i = (127 << mantissabits) | (orig_i & ~(~0 << mantissabits));
  const float frac_f = *((float*)&frac_i);
  const float log_f = frac_f + (expo - 128);  // - bias - 1
  return log_f;
}


static __device__ inline float d_REL_pow2approxf(const float log_f)
{
  const int mantissabits = 23;
  const float biased = log_f + 127;
  const int expo = biased;
  const float frac_f = biased - (expo - 1);
  const int frac_i = *((int*)&frac_f);
  const int exp_i = (expo << mantissabits) | (frac_i & ~(~0 << mantissabits));
  const float recon_f = *((float*)&exp_i);
  return recon_f;
}


// copy (len) bytes from global memory (source) to shared memory (destination) using separate shared memory buffer (temp)
// destination and temp must we word aligned, accesses up to CS + 3 bytes in temp
static inline __device__ void g2s(void* const __restrict__ destination, const void* const __restrict__ source, const int len, void* const __restrict__ temp)
{
  const int tid = threadIdx.x;
  const byte* const __restrict__ input = (byte*)source;
  if (len < 128) {
    byte* const __restrict__ output = (byte*)destination;
    if (tid < len) output[tid] = input[tid];
  } else {
    const int nonaligned = (int)(size_t)input;
    const int wordaligned = (nonaligned + 3) & ~3;
    const int linealigned = (nonaligned + 127) & ~127;
    const int bcnt = wordaligned - nonaligned;
    const int wcnt = (linealigned - wordaligned) / 4;
    int* const __restrict__ out_w = (int*)destination;
    if (bcnt == 0) {
      const int* const __restrict__ in_w = (int*)input;
      byte* const __restrict__ out = (byte*)destination;
      if (tid < wcnt) out_w[tid] = in_w[tid];
      for (int i = tid + wcnt; i < len / 4; i += TPB) {
        out_w[i] = in_w[i];
      }
      if (tid < (len & 3)) {
        const int i = len - 1 - tid;
        out[i] = input[i];
      }
    } else {
      const int offs = 4 - bcnt;  //(4 - bcnt) & 3;
      const int shift = offs * 8;
      const int rlen = len - bcnt;
      const int* const __restrict__ in_w = (int*)&input[bcnt];
      byte* const __restrict__ buffer = (byte*)temp;
      byte* const __restrict__ buf = (byte*)&buffer[offs];
      int* __restrict__ buf_w = (int*)&buffer[4];  //(int*)&buffer[(bcnt + 3) & 4];
      if (tid < bcnt) buf[tid] = input[tid];
      if (tid < wcnt) buf_w[tid] = in_w[tid];
      for (int i = tid + wcnt; i < rlen / 4; i += TPB) {
        buf_w[i] = in_w[i];
      }
      if (tid < (rlen & 3)) {
        const int i = len - 1 - tid;
        buf[i] = input[i];
      }
      __syncthreads();
      buf_w = (int*)buffer;
      for (int i = tid; i < (len + 3) / 4; i += TPB) {
        out_w[i] = __funnelshift_r(buf_w[i], buf_w[i + 1], shift);
      }
    }
  }
}


static __device__ int g_chunk_counter;


static __global__ void d_reset()
{
  g_chunk_counter = 0;
}


static __device__ inline void d_iQREL_4(int& csize, byte in [CS], byte out [CS], const float errorbound)
{
  using ftype = float;
  using itype = int;
  const int size = csize / sizeof(ftype);
  const int tid = threadIdx.x;

  const int mantissabits = 23;
  const itype signexpomask = ~0 << mantissabits;

  const ftype log2eb = 2 * d_REL_log2approxf(1 + errorbound);

  itype* const data_in_i = (itype*)in;
  itype* const data_out_i = (itype*)out;

  for (int i = tid; i < size; i += TPB) {
    itype val = (data_in_i[i] + 1) ^ signexpomask;
    if (((val & signexpomask) == signexpomask) && ((val & ~signexpomask) != 0)) {  // is encoded value
      if (val == (signexpomask | 1)) {
        val = 0;
      } else {
        const itype dec = ((val & ~signexpomask) >> 1) - 1;
        const itype bin = (dec >> 1) ^ (((dec << 31) >> 31));  // TCMS decoding
        const ftype abs_recon_f = d_REL_pow2approxf(bin * log2eb);
        const ftype output = (val & 1) ? -abs_recon_f : abs_recon_f;
        val = *((itype*)&output);
      }
    }
    data_out_i[i] = val;
  }
}


#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 800)
static __global__ __launch_bounds__(TPB, 3)
#else
static __global__ __launch_bounds__(TPB, 2)
#endif
void d_decode(const byte* const __restrict__ input, byte* const __restrict__ output, int* const __restrict__ g_outsize)
{
  // allocate shared memory buffer
  __shared__ long long chunk [3 * (CS / sizeof(long long))];
  const int last = 3 * (CS / sizeof(long long)) - 2 - WS;

  // input header
  long long* const head_in = (long long*)input;
  float* const head_in_f = (float*)&head_in[1];
  const int outsize = (int)head_in[0];
  const float errorbound = head_in_f[0];

  // initialize
  const int chunks = (outsize + CS - 1) / CS;  // round up
  unsigned short* const size_in = (unsigned short*)&head_in[2];
  byte* const data_in = (byte*)&size_in[chunks];

  // loop over chunks
  const int tid = threadIdx.x;
  int prevChunkID = 0;
  int prevOffset = 0;
  do {
    // assign work dynamically
    if (tid == 0) chunk[last] = atomicAdd(&g_chunk_counter, 1);
    __syncthreads();  // chunk[last] produced, chunk consumed

    // terminate if done
    const int chunkID = chunk[last];
    const int base = chunkID * CS;
    if (base >= outsize) break;

    // compute sum of all prior csizes (start where left off in previous iteration)
    int sum = 0;
    for (int i = prevChunkID + tid; i < chunkID; i += TPB) {
      sum += (int)size_in[i];
    }
    int csize = (int)size_in[chunkID];
    const int offs = prevOffset + block_sum_reduction(sum, (int*)&chunk[last + 1]);
    prevChunkID = chunkID;
    prevOffset = offs;

    // create the 3 shared memory buffers
    byte* in = (byte*)&chunk[0 * (CS / sizeof(long long))];
    byte* out = (byte*)&chunk[1 * (CS / sizeof(long long))];
    byte* temp = (byte*)&chunk[2 * (CS / sizeof(long long))];

    // load chunk
    g2s(in, &data_in[offs], csize, out);
    byte* tmp = in; in = out; out = tmp;
    __syncthreads();  // chunk produced, chunk[last] consumed

    // decode
    const int osize = min(CS, outsize - base);
    if (csize < osize) {
      byte* tmp;
      tmp = in; in = out; out = tmp;
      d_iRZE_1(csize, in, out, temp);
      __syncthreads();
      tmp = in; in = out; out = tmp;
      d_iBIT_4(csize, in, out, temp);
      __syncthreads();
      tmp = in; in = out; out = tmp;
      d_iDIFFNB_4(csize, in, out, temp);
      __syncthreads();
      tmp = in; in = out; out = tmp;
      d_iQREL_4(csize, in, out, errorbound);
      __syncthreads();
     }

    long long* const output_l = (long long*)&output[base];
    long long* const out_l = (long long*)out;
    for (int i = tid; i < osize / 8; i += TPB) {
      output_l[i] = out_l[i];
    }
    const int extra = osize % 8;
    if (tid < extra) output[base + osize - extra + tid] = out[osize - extra + tid];
  } while (true);

  if ((blockIdx.x == 0) && (tid == 0)) {
    *g_outsize = outsize;
  }
}


struct GPUTimer
{
  cudaEvent_t beg, end;
  GPUTimer() {cudaEventCreate(&beg); cudaEventCreate(&end);}
  ~GPUTimer() {cudaEventDestroy(beg); cudaEventDestroy(end);}
  void start() {cudaEventRecord(beg, 0);}
  double stop() {cudaEventRecord(end, 0); cudaEventSynchronize(end); float ms; cudaEventElapsedTime(&ms, beg, end); return 0.001 * ms;}
};


static void CheckCuda(const int line)
{
  cudaError_t e;
  cudaDeviceSynchronize();
  if (cudaSuccess != (e = cudaGetLastError())) {
    fprintf(stderr, "CUDA error %d on line %d: %s\n\n", e, line, cudaGetErrorString(e));
    throw std::runtime_error("LC error");
  }
}


int main(int argc, char* argv [])
{
  printf("PFPL GPU Single-Precision REL Decompressor\n");
  printf("Copyright 2025 Texas State University\n\n");

  // read input from file
  if (argc < 3) {printf("USAGE: %s compressed_file_name decompressed_file_name\n\n", argv[0]);  throw std::runtime_error("LC error");}

  // read input file
  FILE* const fin = fopen(argv[1], "rb");
  int pre_size = 0;
  const int pre_val = fread(&pre_size, sizeof(pre_size), 1, fin); assert(pre_val == sizeof(pre_size));
  fseek(fin, 0, SEEK_END);
  const int hencsize = ftell(fin);  assert(hencsize > 0);
  byte* const hencoded = new byte [pre_size];
  fseek(fin, 0, SEEK_SET);
  const int insize = fread(hencoded, 1, hencsize, fin);  assert(insize == hencsize);
  fclose(fin);
  printf("encoded size: %d bytes\n", insize);

  // get GPU info
  cudaSetDevice(0);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  if ((deviceProp.major == 9999) && (deviceProp.minor == 9999)) {fprintf(stderr, "ERROR: no CUDA capable device detected\n\n"); throw std::runtime_error("LC error");}
  const int SMs = deviceProp.multiProcessorCount;
  const int mTpSM = deviceProp.maxThreadsPerMultiProcessor;
  const int blocks = SMs * (mTpSM / TPB);
  CheckCuda(__LINE__);

  // allocate GPU memory
  byte* ddecoded;
  cudaMallocHost((void **)&ddecoded, pre_size);
  byte* d_encoded;
  cudaMalloc((void **)&d_encoded, insize);
  cudaMemcpy(d_encoded, hencoded, insize, cudaMemcpyHostToDevice);
  byte* d_decoded;
  cudaMalloc((void **)&d_decoded, pre_size);
  int* d_decsize;
  cudaMalloc((void **)&d_decsize, sizeof(int));
  CheckCuda(__LINE__);

  // time GPU decoding
  GPUTimer dtimer;
  int ddecsize = 0;
  for (int i = 0; i < NUM_RUNS; i++) {
    cudaDeviceSynchronize();
    dtimer.start();

    d_reset<<<1, 1>>>();
    d_decode<<<blocks, TPB>>>(d_encoded, d_decoded, d_decsize);

    cudaDeviceSynchronize();
    double runtime = dtimer.stop();

    if (i < NUM_RUNS - 1) {
      cudaMemset(d_decsize, 0, sizeof(int));
      cudaMemset(d_decoded, 0, pre_size);
    }
    CheckCuda(__LINE__);
    printf("lc decomp ecltime, %12.9f\n", runtime);
  }

  // get decoded GPU result
  cudaMemcpy(&ddecsize, d_decsize, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(ddecoded, d_decoded, ddecsize, cudaMemcpyDeviceToHost);
  printf("decoded size: %d bytes\n", ddecsize);
  printf("compression ratio: %.2f\n", 1.0 * ddecsize / insize);
  CheckCuda(__LINE__);

  // write to file
  FILE* const fout = fopen(argv[2], "wb");
  fwrite(ddecoded, 1, ddecsize, fout);
  fclose(fout);

  // clean up GPU memory
  cudaFree(d_encoded);
  cudaFree(d_decoded);
  cudaFree(d_decsize);
  CheckCuda(__LINE__);

  // clean up
  cudaFreeHost(ddecoded);
  return 0;
}
