#ifndef BNDMQ2_CUH
#define BNDMQ2_CUH

#include "include/define.cuh"

__global__
void backward_nondeterministic_dawg_qgram(
   unsigned char *text, int text_size, 
   unsigned char *pattern, int pattern_size,
   unsigned int *B, unsigned int M, int stride_length, int *match);

__global__
void backward_nondeterministic_dawg_qgram_large(
   unsigned char *text, int text_size, 
   unsigned char *pattern, int pattern_size,
   unsigned int *B, unsigned int M, int stride_length, int *match);

#endif
