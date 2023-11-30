#ifndef FNDM_CUH
#define FNDM_CUH

#include "include/define.cuh"

__global__
void forward_nondeterministic_dawg( 
        unsigned char *text, int text_size, 
        unsigned char *pattern, int pattern_size,
        int *B, int stride_length, int *match);

__global__
void forward_nondeterministic_dawg_large(
    unsigned char *text, int text_size, 
    unsigned char *pattern, int pattern_size,
    int *B, int stride_length, int *match);

#endif
