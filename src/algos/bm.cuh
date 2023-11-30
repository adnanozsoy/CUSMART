#ifndef BM_CUH
#define BM_CUH

#include "include/define.cuh"

#define XSIZE 200

__global__ 
void boyer_moore(unsigned char *text, unsigned long text_size,
                            unsigned char *pattern, int pattern_size,int *bmGs, 
                            int *bmBc, int search_len, int *match);

void preBmBc(unsigned char *x, int m, int bmBc[]);
void suffixes(unsigned char *x, int m, int *suff);
void preBmGs(unsigned char *x, int m, int bmGs[]);

#endif


