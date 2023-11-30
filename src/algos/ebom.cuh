#ifndef EBOM_CUH
#define EBOM_CUH

#include "include/define.cuh"

#define UNDEFINED -1

__global__
void ebom(unsigned char *text, unsigned long text_size,
	  unsigned char *pattern, int pattern_size,int **FT,
	  int **trans, int search_len, int *match);

#endif
