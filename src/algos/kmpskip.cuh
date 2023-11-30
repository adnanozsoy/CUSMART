#ifndef KMPSKIP_CUH
#define KMPSKIP_CUH

#include "include/define.cuh"

__global__
void kmpskip(unsigned char *text, unsigned long text_size, unsigned char *pattern,
	     int pattern_size, int *kmpNext, int *list, int *mpNext,
	     int *z, int search_len, int *match);

__host__ void preKmp(unsigned char *pattern, int pattern_size, int kmpNext[]);
__host__ void preMp(unsigned char *pattern, int pattern_size, int mpNext[]);
__host__ __device__ int attempt(unsigned char *text, unsigned char *pattern, int pattern_size, int start, int wall);

#endif
