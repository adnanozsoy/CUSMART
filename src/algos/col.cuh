
#ifndef COL_CUH
#define COL_CUH


__global__
void colussi(  unsigned char *text, int text_size, 
               unsigned char *pattern, int pattern_size,
               int nd, int *h, int *next, int *shift, int stride_length, int *match);

__host__
int preColussi(unsigned char *pattern, int pattern_size, 
   int *h, int *next, int *shift);

#endif





