#ifndef MP_CUH
#define MP_CUH

__host__
void pre_morris_pratt(	unsigned char *pattern, int pattern_size, 
						int *shift_array);

__global__
void morris_pratt(unsigned char *text, unsigned long text_size, 
                  unsigned char *pattern, int pattern_size, 
                  int *shift_array, int stride_length, int *match);

#endif