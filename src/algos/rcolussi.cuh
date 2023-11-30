#ifndef COL_CUH
#define COL_CUH

#include "include/define.cuh"

__global__
void reverse_colussi(unsigned char *text, int text_size, 
               unsigned char *pattern, int pattern_size,
               int *h, int *rcBc, int *rcGs, int stride_length, int *match);

__host__
void preRc(unsigned char *x, int m, int h[], int rcBc[], int rcGs[]);

#endif
