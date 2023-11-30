#ifndef SEBOM_CUH
#define SEBOM_CUH

#include "include/define.cuh"

#define UNDEFINED -1
#define FT(i,j)  LAMBDA[(i<<8) + j]

__global__
void sebom(unsigned char *text, unsigned long text_size,
	  unsigned char *pattern, int pattern_size,int *LAMBDA,
	  int **trans, int search_len, int *match);

#endif