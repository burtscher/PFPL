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


static inline float d_REL_log2approxf(const float orig_f)
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


static inline float d_REL_pow2approxf(const float log_f)
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


struct CPUTimer
{
  timeval beg, end;
  CPUTimer() {}
  ~CPUTimer() {}
  void start() {gettimeofday(&beg, NULL);}
  double stop() {gettimeofday(&end, NULL); return end.tv_sec - beg.tv_sec + (end.tv_usec - beg.tv_usec) / 1000000.0;}
};


static inline void h_QREL_4(int& csize, byte in [CS], byte out [CS], const float errorbound, const float threshold)
{
  using ftype = float;
  using itype = int;
  const int size = csize / sizeof(ftype);

  const int mantissabits = 23;
  const itype signexpomask = ~0 << mantissabits;
  const itype maxbin = (1 << (mantissabits - 2)) - 1;  // leave 2 bits for 2 signs (plus one element)

  const ftype log2eb = 2 * d_REL_log2approxf(1 + errorbound);
  const ftype inv_log2eb = 1 / log2eb;

  itype* const data_out_i = (itype*)out;
  itype* const data_in_i = (itype*)in;

  #pragma omp parallel for default(none) shared(signexpomask, size, data_out_i, data_in_i, errorbound, log2eb, inv_log2eb, threshold, maxbin, mantissabits)
  for (int i = 0; i < size; i ++) {
    const itype orig_i = data_in_i[i];
    const itype abs_orig_i = orig_i & 0x7fff'ffff;
    const ftype abs_orig_f = *((ftype*)&abs_orig_i);
    itype output = orig_i;
    const int expo = (orig_i >> mantissabits) & 0xff;
    if (expo == 0) {  // zero or de-normal values
      if (abs_orig_i == 0) {  // zero
        output = signexpomask | 1;
      }
    } else {
      if (expo == 0xff) {  // INF or NaN
        if (((orig_i & signexpomask) == signexpomask) && ((orig_i & ~signexpomask) != 0)) {  // negative NaN
          output = abs_orig_i;  // make positive NaN
        }
      } else {  // normal value
        const ftype log_f = d_REL_log2approxf(abs_orig_f);
        const ftype scaled = log_f * inv_log2eb;
        itype bin = (itype)roundf(scaled);
        const ftype abs_recon_f = d_REL_pow2approxf(bin * log2eb);
        const ftype lower = abs_orig_f / (1 + errorbound);
        const ftype upper = abs_orig_f * (1 + errorbound);
        if (!((bin >= maxbin) || (bin <= -maxbin) || (abs_orig_f >= threshold) || (abs_recon_f < lower) || (abs_recon_f > upper) || (abs_recon_f == 0) || !std::isfinite(abs_recon_f))) {
          bin = (bin << 1) ^ (bin >> 31);  // TCMS encoding
          bin = (bin + 1) << 1;
          if (orig_i < 0) bin |= 1;  // include sign
          output = signexpomask | bin;  // 'sign' and 'exponent' fields are all ones, 'mantissa' is non-zero (looks like a negative NaN)
        }
      }
    }
    data_out_i[i] = (output ^ signexpomask) - 1;
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
      h_QREL_4(csize, in, out, errorbound, threshold);
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
  printf("PFPL CPU Single-Precision REL Compressor\n");
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
