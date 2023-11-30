#ifndef FJS_CUH
#define FJS_CUH

#include "include/define.cuh"

__global__
void fjs(unsigned char *text, unsigned long text_size, unsigned char *pattern,
	     int pattern_size, int *qsbc, int *kmp, int search_len, int *match);

__host__ void preKmpFJS(unsigned char *pattern, int pattern_size, int kmpNext[]);
__host__ void preQsBcFJS(unsigned char *pattern, int pattern_size, int qbc[]);

#endif
