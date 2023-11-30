#ifndef BNDMQ4_CUH
#define BNDMQ4_CUH

#include "include/define.cuh"

__global__
void backward_nondeterministic_dawg_qgram4(
   unsigned char *text, int text_size, 
   unsigned char *pattern, int pattern_size,
   unsigned int *B, unsigned int M, int stride_length, int *match);

__global__
void backward_nondeterministic_dawg_qgram4_large(
   unsigned char *text, int text_size, 
   unsigned char *pattern, int pattern_size,
   unsigned int *B, unsigned int M, int stride_length, int *match);

#endif
