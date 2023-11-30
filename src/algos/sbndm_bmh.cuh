#ifndef SBNDM_BMH_CUH
#define SBNDM_BMH_CUH

#include "include/define.cuh"

__global__
void simplified_backward_nondeterministic_dawg_horspool_shift(
   unsigned char *text, int text_size, 
   unsigned char *pattern, int pattern_size,
   unsigned int *B, int *hbc, int shift, int stride_length, int *match);

__global__
void simplified_backward_nondeterministic_dawg_horspool_shift_large(
   unsigned char *text, int text_size, 
   unsigned char *pattern, int pattern_size,
   unsigned int *B, int *hbc, int shift, int stride_length, int *match);

#endif
