
#ifndef BF_CUH
#define BF_CUH

__global__ 
void brute_force(	unsigned char *text, unsigned long text_size, 
					unsigned char *pattern, int pattern_size, int *match);

#endif


