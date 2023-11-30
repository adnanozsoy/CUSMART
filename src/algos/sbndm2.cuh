#ifndef SBNDM2_CUH
#define SBNDM2_CUH

#include "include/define.cuh"

__global__
void simplified_backward_nondeterministic_dawg_unrolled(
    unsigned char *text, unsigned long text_size,
    unsigned char *pattern, int pattern_size,
    unsigned int *B, int shift, int stride_length, int *match);

__global__
void simplified_backward_nondeterministic_dawg_unrolled_large(
    unsigned char *text, unsigned long text_size,
    unsigned char *pattern, int pattern_size,
    unsigned int *B, int shift, int stride_length, int *match);

#endif
