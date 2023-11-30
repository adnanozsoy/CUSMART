#ifndef TUNEDBM_CUH
#define TUNEDBM_CUH

#include "include/define.cuh"

__global__ 
void tuned_boyer_moore(unsigned char *text, unsigned long text_size,
                            unsigned char *pattern, int pattern_size, int *bmBc, 
                            int shift, int search_len, int *match);

void preBmBc(unsigned char *x, int m, int bmBc[]);

#endif


