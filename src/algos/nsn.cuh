#ifndef NSN_CUH
#define NSN_CUH

__global__ 
void not_so_naive(unsigned char *text, unsigned long text_size,
		  unsigned char *pattern, int pattern_size, int k, 
		  int ell, int search_len, int *match);

#endif
