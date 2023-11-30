#ifndef TSW_CUH
#define TSW_CUH

#include "include/define.cuh"

__global__
void two_sliding_window(unsigned char *text, unsigned long text_size,
		 unsigned char *pattern, int pattern_size, int **brBc_left,
		 int **brBc_right, int search_len, int *match);

void preBrBcTSW(unsigned char *pattern, int pattern_size, int brBc[SIGMA][SIGMA]);

#endif