
#ifndef BFBS_CUH
#define BFBS_CUH

#define SHARED_MEMORY_SIZE 6144

__global__ 
void brute_force_block_shared(	unsigned char *text, unsigned long text_size, 
								unsigned char *pattern, int pattern_size, 
								int stride_length, int *match);

#endif


