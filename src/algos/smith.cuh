#ifndef SMITH_CUH
#define SMITH_CUH

#include "include/define.cuh"

#define MAX(a,b) ((a) > (b) ? (a) : (b))

__global__
void smith(unsigned char *text, unsigned long text_size, unsigned char *pattern, int pattern_size,
	      unsigned char bmBc[], unsigned char qsBc[], int stride_length, int *match);

void preBmBcSMITH(unsigned char *pattern, int pattern_size, unsigned char bmBc[]);
void preQsBcSMITH(unsigned char *pattern, int pattern_size, unsigned char qsBc[]);

#endif
