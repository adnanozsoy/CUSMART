#include "nsn.cuh"

__global__
void not_so_naive(unsigned char *text, unsigned long text_size,
                  unsigned char *pattern, int pattern_size, int k,
                  int ell, int search_len, int *match){
  
      int i;
      unsigned long thread_id = blockDim.x * blockIdx.x + threadIdx.x;
      unsigned long start_inx = thread_id * search_len;
      
      unsigned long boundary = start_inx + search_len;
      boundary = boundary > text_size ? text_size : boundary;
      unsigned long j = start_inx;
      
      while (j < boundary && j <= text_size - pattern_size) {
	    i = 0;
	    if (pattern[1] != text[j + 1])
	          j += k;
	    else {
	          if(pattern_size == 1 || pattern_size == 2){
		        if (pattern[0] == text[j]) match[j] = 1;
		  }
		  else {
		        while (i < pattern_size - 2 && pattern[i + 2] == text[j + i + 2] &&
			       pattern[0] == text[j]) i++;
			if(i == pattern_size - 2) match[j] = 1; 
		  }
		  j += ell;
	    }
      }
}
