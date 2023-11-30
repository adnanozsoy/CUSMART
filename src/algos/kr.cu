#include "kr.cuh"

//this function only return 1 if equal otherwise return 0
__device__ static int isEqual(unsigned char *str_a, unsigned char *str_b, int len){
	int i = 0;
	while (str_a[i] == str_b[i]) i++;
	return i == len;
}
  
 

__global__ void karp_rabin(unsigned char *text, unsigned long text_size,
	unsigned char *pattern, int pattern_size, int hash_factor, 
	int hpattern, int stride_length, int *match)
{
    int upper_limit;
    int idx = (threadIdx.x + blockIdx.x*blockDim.x) * stride_length;
    if ( idx > text_size - pattern_size ) return;

    if (idx <= text_size - pattern_size - stride_length)
        upper_limit = idx + stride_length;
    else
        upper_limit = text_size - pattern_size;
	unsigned long i, j = idx;
	
	int htext = 0;	
	for (i = idx; i < idx + pattern_size; ++i) {				
		htext = ((htext<<1) + text[i]);
	}


	while (j < upper_limit ) {		
		if (hpattern == htext && isEqual(pattern, text + j, pattern_size)) 
			match[j] = 1;
		//#define REHASH(a, b, h) ((((h) - (a)*d) << 1) + (b))
		htext = ((((htext) - (text[j])*hash_factor) << 1) + (text[j + pattern_size]));
		++j;
	}
}
