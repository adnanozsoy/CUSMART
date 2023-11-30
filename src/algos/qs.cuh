#ifndef QS_CUH
#define QS_CUH

#include "include/define.cuh"

__host__
void preQsBc(unsigned char *pattern, int pattern_size, int *qbc);

__global__
void quicksearch( unsigned char *text, int text_size,
                  unsigned char *pattern, int pattern_size,
                  int *bmBc, int stride_length, int *match);

#endif