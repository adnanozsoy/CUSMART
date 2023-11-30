
#ifndef SMOA_CUH
#define SMOA_CUH

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

__global__ 
void string_matching_ordered_alphabet(unsigned char *text, unsigned long text_size,
	unsigned char *pattern, int pattern_size, int search_len, int *match);

#endif


