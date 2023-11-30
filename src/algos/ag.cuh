#ifndef AG_CUH
#define AG_CUH

#include "include/define.cuh"

__host__
void preBmBcAG(unsigned char *pattern, int pattern_size, int *bmBc);
 
 __host__
void suffixesAG(unsigned char *pattern, int pattern_size, int *suff);

__host__
void preBmGsAG(unsigned char *pattern, int pattern_size, int *bmGs, int *suff);

__global__
void ag(unsigned char *text, int text_size, 
        unsigned char *pattern, int pattern_size,
        int *bmGs, int *bmBc, int *suff,
        int stride_length, int *match);

#endif
