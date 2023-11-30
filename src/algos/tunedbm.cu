#include "tunedbm.cuh"
#include "bm.cuh"

//this function only return 1 if equal otherwise return 0
__device__ int isEqual2(unsigned char *str_a, unsigned char *str_b, int len){
	int equal = 1;
	unsigned i = 0;
	while ((i < len) && equal){		
		if (str_a[i] != str_b[i]){			
			equal = 0;
		}
		i++;
	}
	return equal;
}


__global__ void tuned_boyer_moore(unsigned char *text, unsigned long text_size,
	unsigned char *pattern, int pattern_size, int *bmBc, int shift, 
	int search_len, int *match)
{
	unsigned long thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned long start_inx = thread_id * search_len;
	
	unsigned long boundary = start_inx + search_len;
	boundary = boundary > text_size ? text_size : boundary;
	int k;
	unsigned long j = start_inx;

	while (j < boundary + pattern_size - 1 && j <= text_size - pattern_size) {
		k = bmBc[text[j + pattern_size - 1]];		
		while (k != 0 && j < boundary + pattern_size - 1) {
			j += k;
			if(j > text_size - pattern_size)
				k = 0;
			else
				k = bmBc[text[j + pattern_size - 1]];
		}
		
		if (k == 0 && isEqual2(pattern, text + j, pattern_size - 1) && j <= text_size - pattern_size)			
				match[j] = 1;
		if(k == 0)
			j += shift;
	}
}
