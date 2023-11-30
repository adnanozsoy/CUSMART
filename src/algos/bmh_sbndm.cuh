#ifndef BMH_SBNDM_CUH
#define BMH_SBNDM_CUH

#include "include/define.cuh"

__global__
void horspool_with_bndm(
    unsigned char *text, int text_size,
    unsigned char *pattern, int pattern_size,
    unsigned int *B, int *hbc, int shift, int stride_length, int *match);

__global__
void horspool_with_bndm_large(
   unsigned char *text, int text_size, 
   unsigned char *pattern, int pattern_size,
   unsigned int *B, int *hbc, int shift, int stride_length, int *match);

#endif
