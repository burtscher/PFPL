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


static inline void h_iQABS_4(int& csize, byte in [CS], byte out [CS], const float errorbound)
{
  using ftype = float;
  using itype = int;
  const int size = csize / sizeof(ftype);

  const int mantissabits = 23;
  const ftype eb2 = 2 * errorbound;

  const itype* const data_i = (itype*)in;
  ftype* const data_f = (ftype*)out;

  for (int i = 0; i < size; i ++) {
    itype bin = data_i[i];
    ftype val;
    if ((0 <= bin) && (bin < (1 << mantissabits))) {  // is encoded value
      bin = (bin >> 1) ^ (((bin << 31) >> 31));  // TCMS decoding
      val = bin * eb2;
    } else {
      val = *((ftype*)&bin);
    }
    data_f[i] = val;
  }
}


static void h_decode(const byte* const __restrict__ input, byte* const __restrict__ output, int& outsize)
{
  // input header
  long long* const head_in = (long long*)input;
  float* const head_in_f = (float*)&head_in[1];
  outsize = (int)head_in[0];
  const float errorbound = head_in_f[0];

  // initialize
  const int chunks = (outsize + CS - 1) / CS;  // round up
  unsigned short* const size_in = (unsigned short*)&head_in[2];
  byte* const data_in = (byte*)&size_in[chunks];
  int* const start = new int [chunks];

  // convert chunk sizes into starting positions
  int pfs = 0;
  for (int chunkID = 0; chunkID < chunks; chunkID++) {
    start[chunkID] = pfs;
    pfs += (int)size_in[chunkID];
  }

  // process chunks in parallel
  #pragma omp parallel for schedule(dynamic, 1)
  for (int chunkID = 0; chunkID < chunks; chunkID++) {
    // load chunk
    long long chunk1 [CS / sizeof(long long)];
    long long chunk2 [CS / sizeof(long long)];
    byte* in = (byte*)chunk1;
    byte* out = (byte*)chunk2;
    const int base = chunkID * CS;
    const int osize = std::min(CS, outsize - base);
    int csize = size_in[chunkID];
    if (csize == osize) {
      // simply copy
      memcpy(&output[base], &data_in[start[chunkID]], osize);
    } else {
      // decompress
      memcpy(out, &data_in[start[chunkID]], csize);

      // decode
      std::swap(in, out);
      h_iRZE_1(csize, in, out);
      std::swap(in, out);
      h_iBIT_4(csize, in, out);
      std::swap(in, out);
      h_iDIFFNB_4(csize, in, out);
      std::swap(in, out);
      h_iQABS_4(csize, in, out, errorbound);

      memcpy(&output[base], out, csize);
    }
  }

  delete [] start;
}


int main(int argc, char* argv [])
{
  printf("PFPL CPU Single-Precision ABS Decompressor\n");
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

  // allocate CPU memory
  byte* hdecoded = new byte [pre_size];
  int hdecsize = 0;

  CPUTimer ctimer;
  for (int i = 0; i < NUM_RUNS; i++) {
    ctimer.start();

    h_decode(hencoded, hdecoded, hdecsize);

    double runtime = ctimer.stop();
    printf("lc decomp ecltime, %12.9f\n", runtime);
  }

  printf("decoded size: %d bytes\n", hdecsize);
  printf("compression ratio: %.2f\n", 1.0 * hdecsize / insize);

  // write to file
  FILE* const fout = fopen(argv[2], "wb");
  fwrite(hdecoded, 1, hdecsize, fout);
  fclose(fout);

  delete [] hencoded;
  delete [] hdecoded;
  return 0;
}
