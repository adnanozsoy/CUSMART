#ifndef FFS_CUH
#define FFS_CUH

#include "include/define.cuh"

__global__
void forward_fast_search(
    unsigned char *text, int text_size,
    unsigned char *pattern, int pattern_size,
    int *bc, int *gs, int stride_length, int *match);

__host__
void forward_suffix_function(unsigned char *x, int m, int *bm_gs, int s);

#endif
