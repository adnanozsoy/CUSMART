#ifndef HOR_CUH
#define HOR_CUH

#include "include/define.cuh"

__global__
void horspool(unsigned char *text, unsigned long text_size, unsigned char *pattern, int pattern_size,
     		         unsigned char hbc[], int stride_length, int *match);

void pre_horspool(unsigned char *pattern, int pattern_size, unsigned char hbc[]);

#endif