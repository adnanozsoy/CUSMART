#ifndef SBNDMQ2_CUH
#define SBNDMQ2_CUH

#include "include/define.cuh"

__global__
void simplified_backward_nondeterministic_dawg_qgram(
   unsigned char *text, int text_size, 
   unsigned char *pattern, int pattern_size,
   unsigned int *B, int mq, int mMinusq, int shift,
   int stride_length, int *match);

__global__
void simplified_backward_nondeterministic_dawg_qgram_large(
   unsigned char *text, int text_size, 
   unsigned char *pattern, int pattern_size,
   unsigned int *B, int mq, int mMinusq, int shift,
   int stride_length, int *match);

#endif