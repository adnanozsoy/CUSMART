#ifndef BLIM_CUH
#define BLIM_CUH

#include "include/define.cuh"

__global__
void bit_parallel_length_invariant_matcher(
    unsigned char *text, int text_size,
    unsigned char *pattern, int pattern_size,
    unsigned int *ScanOrder, unsigned int *MScanOrder,
    unsigned long *MM, unsigned int *shift,
    int stride_length, int *match);

#endif


