#ifndef SSABS_CUH
#define SSABS_CUH

#include "include/define.cuh"

__host__
void preQsBcSSABS(unsigned char *pattern, int pattern_size, int qbc[]);

__global__
void ssabs( 
        unsigned char *text, int text_size, 
        unsigned char *pattern, int pattern_size,
        int *qsBc, unsigned char firstCh, unsigned char lastCh, int stride_length, int *match);

#endif
