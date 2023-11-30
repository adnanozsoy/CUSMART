#include "tbm.cuh"

__global__ void turbo_boyer_moore(unsigned char *text, unsigned long text_size,
	unsigned char *pattern, int pattern_size,int *bmGs, int *bmBc, 
	int search_len, int *match)
{
	unsigned long thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned long start_inx = thread_id * search_len;
	
	unsigned long boundary = start_inx + search_len;
	boundary = boundary > text_size ? text_size : boundary;
	int bcShift, i, shift, u, v, turboShift;
	unsigned long j = start_inx;	
	
	u = 0;
    shift = pattern_size;

	while (j < boundary && j <= text_size - pattern_size) {
		 i = pattern_size - 1;		
		 while (i >= 0 && pattern[i] == text[i + j]) {
			--i;
			if (u != 0 && i == pattern_size - 1 - shift)
				i -= u;
		 }
		 
		 if (i < 0) {
			 match[j] = 1;
			 shift = bmGs[0];
			 u = pattern_size - shift;
		 }
		 else{
			v = pattern_size - 1 - i;
			turboShift = u - v;
			bcShift = bmBc[text[i + j]] - pattern_size + 1 + i;
			shift = MAX(turboShift, bcShift);
			shift = MAX(shift, bmGs[i]);
			
			if (shift == bmGs[i])
				u = MIN(pattern_size - shift, v);
            else {
				if (turboShift < bcShift)
					shift = MAX(shift, u + 1);
				u = 0;
            }
		
		 }
		 j += shift;
	}
}

