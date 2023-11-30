#ifndef RAITA_CUH
#define RAITA_CUH

#include "include/define.cuh"

__global__
void raita(unsigned char *text, unsigned long text_size, unsigned char *pattern, int pattern_size, unsigned char bmBc[],
	   char firstCh, char middleCh, char lastCh, int stride_length, int *match);

void preBmBcRAITA(unsigned char *pattern, int pattern_size, unsigned char bmBc[]);

#endif
