#ifndef LBNDM_CUH
#define LBNDM_CUH

#include "include/define.cuh"

__global__
void long_backward_nondeterministic_dawg(
    unsigned char *text, int text_size, 
    unsigned char *pattern, int pattern_size,
    int *B, int k, int m1, int m2, int rmd, int stride_length, int *match);

#endif
