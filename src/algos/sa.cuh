#ifndef SA_CUH
#define SA_CUH

#include "include/define.cuh"

__global__
void shift_and(unsigned char *text, unsigned long text_size, unsigned char *pattern, int pattern_size,
	       unsigned int *S, unsigned int D, unsigned int F, int stride_length, int *match);

__global__
void shift_and_large(unsigned char *text, unsigned long text_size, unsigned char *pattern, int pattern_size,
		     unsigned int *S, unsigned int D, unsigned int F, int stride_length, int *match);

void preSA(unsigned char *pattern, int pattern_size, unsigned int *S);

#endif
