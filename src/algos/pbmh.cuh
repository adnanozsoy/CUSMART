#ifndef PBMH_CUH
#define PBMH_CUH

#include "include/define.cuh"

__global__ 
void bmh_prob(unsigned char *text, unsigned long text_size,
	      unsigned char *pattern, int pattern_size,int *hbc, 
	      int *v, int search_len, int *match);

#endif
