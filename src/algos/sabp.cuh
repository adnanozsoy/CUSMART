#ifndef SABP_CUH
#define SABP_CUH

#include "include/define.cuh"

__global__
void small_alphabet_bit_parallel(
    unsigned char *text, int text_size,
    unsigned char *pattern, int pattern_size,
    unsigned int *T, unsigned int mask, unsigned int mask2, int search_len, int *match);

__global__
void small_alphabet_bit_parallel_large(
   unsigned char *text, int text_size, 
   unsigned char *pattern, int pattern_size,
   unsigned int *T, unsigned int mask, unsigned int mask2, int search_len, int *match);

__device__
int pow2(int n);

__device__
int mylog2(int unsigned n);

#endif
