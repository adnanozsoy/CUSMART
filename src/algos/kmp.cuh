#ifndef kMP_CUH
#define kMP_CUH

__host__
void pre_knuth_morris_pratt(unsigned char *pattern, int pattern_size, 
							int *shift_array);

__global__
void knuth_morris_pratt(unsigned char *text, unsigned long text_size, 
                  		unsigned char *pattern, int pattern_size, 
                  		int *shift_array, int stride_length, int *match);

#endif