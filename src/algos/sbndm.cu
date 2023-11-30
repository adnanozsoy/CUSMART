#include "sbndm.cuh"

__global__ void simplified_backward_nondeterministic_dawg_matching(unsigned char *text, unsigned long text_size,
			   unsigned char *pattern, int pattern_size, unsigned int *B,
			   int shift, int search_len, int *match) {
	
	unsigned long thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned long start_inx = thread_id * search_len;

	unsigned long boundary = start_inx + search_len + pattern_size - 1;	
	boundary = boundary > text_size ? text_size : boundary;

	
	int j,i, first;
    unsigned int D;
    
	i = start_inx + pattern_size;
	while (i <= boundary && i < text_size) {
	  D = B[text[i]];
	  j = i-1; 
	  first = i-pattern_size+1;
	  while (1) {
		 D = (D << 1) & B[text[j]];
		 if (!((j-first) && D)) break;
		 j--;
	  }
	  if (D != 0) {
		 if (i > boundary && i > text_size) return;
		 match[first] = 1;
		 i += shift;
	  } 
	  else {
		 i = j + pattern_size;
	  }
	}
}


