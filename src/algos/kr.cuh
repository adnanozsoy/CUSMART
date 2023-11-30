
#ifndef KR_CUH
#define KR_CUH

__global__ 
void karp_rabin(unsigned char *text, unsigned long text_size,
	unsigned char *pattern, int pattern_size,int hash_factor, 
	int hpattern, int search_len, int *match);

#endif


