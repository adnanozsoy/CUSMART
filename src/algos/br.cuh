#ifndef BR_CUH
#define BR_CUH

#include "include/define.cuh"

#define getBrBc(p, c) brBc[(p)*SIGMA+(c)]

__global__ 
void berry_ravindran(unsigned char *text, unsigned long text_size,
	unsigned char *pattern, int pattern_size,int *brBc, 
	int search_len, int *match);

void preBrBc(unsigned char *x, int m, int *brBc);

#endif


