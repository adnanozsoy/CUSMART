#ifndef FS_CUH
#define FS_CUH

#include "include/define.cuh"

__global__ 
void fast_search(unsigned char *text, unsigned long text_size,
	unsigned char *pattern, int pattern_size,int *bc, int *gs, 
		int search_len, int *match);

void Pre_GS(unsigned char *x, int m, int bm_gs[]);

#endif


