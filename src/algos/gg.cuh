#ifndef GG_CUH
#define GG_CUH

__global__
void galil_giancarlo(unsigned char *text, int text_size, 
	unsigned char *pattern, int pattern_size,
    int nd, int ell, int *h, int *next, int *shift, 
    int stride_length, int *match);

__host__
int preColussiGG(unsigned char *pattern, int pattern_size,  int *h, int *next, int *shift);

#endif