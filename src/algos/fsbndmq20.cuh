#ifndef FSBNDMQ20_CUH
#define FSBNDMQ20_CUH

#include "include/define.cuh"

__global__
void forward_simplified_bndm_qgram_schar(
    unsigned char *text, int text_size,
    unsigned char *pattern, int pattern_size,
    unsigned int *B, int mm, int sh, int m1,
    int stride_length, int *match);

#endif
