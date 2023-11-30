#ifndef SBNDM_CUH
#define SBNDM_CUH

#include "include/define.cuh"

#define XSIZE 4200

__global__
void simplified_backward_nondeterministic_dawg_matching(unsigned char *text, unsigned long text_size,
			   unsigned char *pattern, int pattern_size, unsigned int *B,
			   int shift, int search_len, int *match);


#endif
