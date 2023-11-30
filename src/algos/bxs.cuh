#ifndef BXS_CUH
#define BXS_CUH

#include "include/define.cuh"
#define Q 1

__global__
void bndm_extended_shift(
    unsigned char *text, int text_size,
    unsigned char *pattern, int pattern_size,
    unsigned int *B, int stride_length, int *match);

#endif
