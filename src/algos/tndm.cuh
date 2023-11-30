#ifndef TNDM_CUH
#define TNDM_CUH

__global__
void two_way_nondeterministic_dawg(
   unsigned char *text, int text_size, 
   unsigned char *pattern, int pattern_size,
   unsigned int *B, unsigned int *restore, int stride_length, int *match);

__global__
void two_way_nondeterministic_dawg_large(
   unsigned char *text, int text_size, 
   unsigned char *pattern, int pattern_size,
   unsigned int *B, unsigned int *restore, int stride_length, int *match);

#endif
