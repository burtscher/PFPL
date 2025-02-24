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


#include <limits>
#include <cmath>
#include <cassert>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <stdexcept>
#include <sys/time.h>
#include "components/h_DIFFNB_4.h"
#include "components/h_BIT_4.h"
#include "components/h_RZE_1.h"


const int NUM_RUNS = 9;


struct CPUTimer
{
  timeval beg, end;
  CPUTimer() {}
  ~CPUTimer() {}
  void start() {gettimeofday(&beg, NULL);}
  double stop() {gettimeofday(&end, NULL); return end.tv_sec - beg.tv_sec + (end.tv_usec - beg.tv_usec) / 1000000.0;}
};


static inline void h_QABS_4(int& csize, byte in [CS], byte out [CS], const float errorbound, const float threshold)
{
  using ftype = float;
  using itype = int;
  const int size = csize / sizeof(ftype);

  const int mantissabits = 23;
  const itype maxbin = 1 << (mantissabits - 1);  // leave 1 bit for sign

  const ftype eb2 = 2 * errorbound;
  const ftype inv_eb2 = 0.5f / errorbound;

  const ftype* const data_f = (ftype*)in;
  itype* const data_i = (itype*)out;

  #pragma omp parallel for default(none) shared(size, data_f, data_i, errorbound, eb2, inv_eb2, threshold, maxbin, mantissabits)
  for (int i = 0; i < size; i ++) {
    const ftype orig_f = data_f[i];
    const ftype scaled = orig_f * inv_eb2;
    const itype bin = (itype)roundf(scaled);
    const ftype recon = bin * eb2;

    itype val;
    if ((bin >= maxbin) || (bin <= -maxbin) || (fabsf(orig_f) >= threshold) || (fabsf(orig_f - recon) > errorbound) || (orig_f != orig_f)) {  // last check is to handle NaNs
      val = *((itype*)&orig_f);
    } else {
      val = (bin << 1) ^ (bin >> 31);  // TCMS encoding, 'sign' and 'exponent' fields are zero
    }
    data_i[i] = val;
  }
}


static void h_encode(const byte* const __restrict__ input, const int insize, byte* const __restrict__ output, int& outsize, const float errorbound, const float threshold)
{
  // initialize
  const int chunks = (insize + CS - 1) / CS;  // round up
  long long* const head_out = (long long*)output;
  unsigned short* const size_out = (unsigned short*)&head_out[2];
  byte* const data_out = (byte*)&size_out[chunks];
  int* const carry = new int [chunks];
  memset(carry, 0, chunks * sizeof(int));

  // process chunks in parallel
  #pragma omp parallel for schedule(dynamic, 1)
  for (int chunkID = 0; chunkID < chunks; chunkID++) {
    // load chunk
    long long chunk1 [CS / sizeof(long long)];
    long long chunk2 [CS / sizeof(long long)];
    byte* in = (byte*)chunk1;
    byte* out = (byte*)chunk2;
    const int base = chunkID * CS;
    const int osize = std::min(CS, insize - base);
    memcpy(out, &input[base], osize);

    // encode chunk
    int csize = osize;
    bool good = true;
    if (good) {
      std::swap(in, out);
      h_QABS_4(csize, in, out, errorbound, threshold);
    }
    if (good) {
      std::swap(in, out);
      h_DIFFNB_4(csize, in, out);
    }
    if (good) {
      std::swap(in, out);
      h_BIT_4(csize, in, out);
    }
    if (good) {
      std::swap(in, out);
      good = h_RZE_1(csize, in, out);
    }

    int offs = 0;
    if (chunkID > 0) {
      do {
        #pragma omp atomic read
        offs = carry[chunkID - 1];
      } while (offs == 0);
      #pragma omp flush
    }
    if (good && (csize < osize)) {
      // store compressed data
      #pragma omp atomic write
      carry[chunkID] = offs + csize;
      size_out[chunkID] = csize;
      memcpy(&data_out[offs], out, csize);
    } else {
      // store original data
      #pragma omp atomic write
      carry[chunkID] = offs + osize;
      size_out[chunkID] = osize;
      memcpy(&data_out[offs], &input[base], osize);
    }
  }

  // output header
  head_out[0] = (long long)insize;
  float* const head_out_f = (float*)&head_out[1];
  head_out_f[0] = errorbound;

  // finish
  outsize = &data_out[carry[chunks - 1]] - output;
  delete [] carry;
}


int main(int argc, char* argv [])
{
  printf("PFPL CPU Single-Precision ABS Compressor\n");
  printf("Copyright 2025 Texas State University\n\n");

  // read input from file
  if (argc < 4 || argc > 5) {printf("USAGE: %s input_file_name compressed_file_name error_bound [threshold]\n\n", argv[0]);  throw std::runtime_error("LC error");}

  FILE* const fin = fopen(argv[1], "rb");
  fseek(fin, 0, SEEK_END);
  const long long fsize = ftell(fin);
  if (fsize <= 0) {fprintf(stderr, "ERROR: input file too small\n\n"); throw std::runtime_error("LC error");}
  if (fsize >= 2147221529) {fprintf(stderr, "ERROR: input file too large\n\n"); throw std::runtime_error("LC error");}
  byte* const input = new byte [fsize];
  fseek(fin, 0, SEEK_SET);
  const int insize = fread(input, 1, fsize, fin);  assert(insize == fsize);
  fclose(fin);
  printf("original size: %d bytes\n", insize);

  // allocate CPU memory
  const int chunks = (insize + CS - 1) / CS;  // round up
  const int maxsize = 3 * sizeof(int) + chunks * sizeof(short) + chunks * CS;
  byte* const hencoded = new byte [maxsize];
  int hencsize = 0;

  const float errorbound = atof(argv[3]);
  const float threshold = (argc == 5) ? atof(argv[4]) : std::numeric_limits<float>::infinity();
  if (threshold < std::numeric_limits<float>::min()) {printf("ERROR: threshold must be a positive, normal, floating-point value\n");  throw std::runtime_error("LC error");}
  CPUTimer ctimer;
  for (int i = 0; i < NUM_RUNS; i++) {
    ctimer.start();

    h_encode(input, insize, hencoded, hencsize, errorbound, threshold);

    double runtime = ctimer.stop();
    printf("lc comp ecltime, %12.9f\n", runtime);
  }

  printf("encoded size: %d bytes\n", hencsize);
  printf("compression ratio: %.2f\n", 1.0 * insize / hencsize);

  // write to file
  FILE* const fout = fopen(argv[2], "wb");
  fwrite(hencoded, 1, hencsize, fout);
  fclose(fout);

  delete [] input;
  delete [] hencoded;
  return 0;
}
