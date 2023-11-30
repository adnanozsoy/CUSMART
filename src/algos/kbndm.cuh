#ifndef KBNDM_CUH
#define KBNDM_CUH

#include "include/define.cuh"

//#define CHAR_BITS 8
//#define WORD_TYPE unsigned int
//#define WORD_BITS (sizeof(WORD_TYPE)*CHAR_BITS)

__global__
void factorized_backward_nondeterministic_dawg( 
			unsigned char *text, int text_size, unsigned char *pattern, int m1,
			int pattern_size, unsigned int M, unsigned int **B, unsigned int *L,
			int stride_length, int *match);

#endif
