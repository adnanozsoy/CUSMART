#include "ts.cuh"

__global__
void tailed_substring(unsigned char *text, int text_size,
        unsigned char *pattern, int pattern_size,
        int stride_length, int *match) { 
        int j, i, k, h, dim; 

   	unsigned long thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned long start_inx = thread_id * stride_length;
	
	unsigned long boundary = start_inx + stride_length;
	boundary = boundary > text_size ? text_size : boundary;
	unsigned long s = start_inx;
   
	/* Searching */ 
	/* phase n.1*/
	i = pattern_size-1; k = pattern_size-1; dim = 1;
	while (s < boundary && s <= text_size-pattern_size && i-dim >= 0) {
          if (pattern[i] != text[s+i]) s++;
	  else {
	    for (j=0; j<pattern_size && pattern[j]==text[s+j]; j++);
	    if (j==pattern_size) match[s] = 1;
	    for (h=i-1; h>=0 && pattern[h]!=pattern[i]; h--);
	    if (dim<i-h) {k=i; dim=i-h;}
	    s+=i-h;
	    i--;
	  }
	}
	
	/* phase n.2 */
	while (s < boundary && s <= text_size - pattern_size) {
          if (pattern[k]!=text[s+k]) s++;
	  else {
	    j=0;
	    while(j<pattern_size && pattern[j]==text[s+j]) j++;
	    if (j==pattern_size) match[s] = 1;
	    s+=dim;
	  }
	}
}
