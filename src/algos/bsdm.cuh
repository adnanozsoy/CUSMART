#ifndef BSDM_CUH
#define BSDM_CUH

#include "include/define.cuh"

__global__
void backward_snr_dawg_matching(
				unsigned char *text, int text_size,
				unsigned char *pattern, int pattern_size, unsigned int *B,
				unsigned int *pos, int len, int start, int stride_length, int *match);

#endif
