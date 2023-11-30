#ifndef SO_CUH
#define SO_CUH

#include "include/define.cuh"

__global__
void shift_or(unsigned char *text, unsigned long text_size, unsigned char *pattern, int pattern_size,
	      unsigned int *S, unsigned int lim, unsigned int D, int stride_length, int *match);

__global__
void shift_or_large(unsigned char *text, unsigned long text_size, unsigned char *pattern,
		    int pattern_size, unsigned int *S, unsigned int lim, unsigned int D, int stride_length, int *match);

int preSo(unsigned char *pattern, int pattern_size, unsigned int *S);

#endif
