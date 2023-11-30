#ifndef FBOM_CUH
#define FBOM_CUH

#include "include/define.cuh"

#define UNDEFINED -1

__global__
void fbom(unsigned char *text, unsigned long text_size,
	  unsigned char *pattern, int pattern_size,int **FT,
	  int **trans, int search_len, int *match);

#endif