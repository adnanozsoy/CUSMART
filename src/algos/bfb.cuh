
#ifndef BFB_CUH
#define BFB_CUH

__global__ 
void brute_force_block(	unsigned char *text, unsigned long text_size, 
						unsigned char *pattern, int pattern_size, 
						int stride_length, int *match);

#endif


