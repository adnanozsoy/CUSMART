#ifndef FSBNDM_CUH
#define FSBNDM_CUH

#include "include/define.cuh"

__global__
void forward_simplified_backward_nondeterministic_dawg_matching(
   unsigned char *text, unsigned long text_size,
   unsigned char *pattern, int pattern_size, 
   unsigned int *B,
   int stride_length, int *match);

__global__
void forward_simplified_backward_nondeterministic_dawg_matching_large(
   unsigned char *text, unsigned long text_size,
   unsigned char *pattern, int pattern_size, 
   unsigned int *B, int stride_length, int *match);
#endif
