#ifndef TVSBS_CUH
#define TVSBS_CUH

#include "include/define.cuh"

__global__
void tvsbs(unsigned char *text, unsigned long text_size,
		 unsigned char *pattern, int pattern_size, int **BrBc,
	         char firstCh, char lastCh, int search_len, int *match);

void TVSBSpreBrBc(unsigned char *pattern, int pattern_size, int brBc[SIGMA][SIGMA]);

#endif
