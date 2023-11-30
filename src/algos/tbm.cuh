#ifndef TBM_CUH
#define TBM_CUH

#include "include/define.cuh"

#define XSIZE 200
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

__global__ 
void turbo_boyer_moore(unsigned char *text, unsigned long text_size,
                            unsigned char *pattern, int pattern_size,int *bmGs, 
                            int *bmBc, int search_len, int *match);


#endif


