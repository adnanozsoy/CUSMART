#ifndef BWW_CUH
#define BWW_CUH

#include "include/define.cuh"

__global__
void bitparallel_wide_window(
    unsigned char *text, unsigned long text_size,
    unsigned char *pattern, int pattern_size, 
    unsigned int *B, unsigned int *C, unsigned int s, unsigned int t,
    int stride_length, int *match);

__global__
void bitparallel_wide_window_large(
    unsigned char *text, unsigned long text_size,
    unsigned char *pattern, int pattern_size, 
    unsigned int *B, unsigned int *C, unsigned int s, unsigned int t,
    int stride_length, int *match);

#endif
