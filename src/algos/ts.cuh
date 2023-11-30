#ifndef TS_CUH
#define TS_CUH

__global__
void tailed_substring( 
        unsigned char *text, int text_size,
	unsigned char *pattern, int pattern_size,
        int stride_length, int *match);

#endif
