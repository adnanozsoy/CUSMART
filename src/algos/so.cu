#include "so.cuh"

__global__ void shift_or(unsigned char *text, unsigned long text_size, unsigned char *pattern,
			 int pattern_size, unsigned int *S, unsigned int lim, unsigned int D, int stride_length, int *match) { 

   	unsigned long thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned long start_inx = thread_id * stride_length;
	unsigned long boundary = start_inx + stride_length + pattern_size;
	boundary = boundary > text_size ? text_size : boundary;
	unsigned long j;

	for (j = start_inx; j < boundary && j < text_size; ++j) {
	      D = (D<<1) | S[text[j]];
	      if (D < lim) match[j - pattern_size + 1] = 1;
	} 
}

__global__ void shift_or_large(unsigned char *text, unsigned long text_size, unsigned char *pattern,
			       int pattern_size, unsigned int *S, unsigned int lim, unsigned int D, int stride_length, int *match) { 
        unsigned int k,h,p_len; 
	unsigned long thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned long start_inx = thread_id * stride_length;
	unsigned long boundary = start_inx + stride_length + pattern_size;
	boundary = boundary > text_size ? text_size : boundary;
	unsigned long j;

   
	p_len = pattern_size;
	pattern_size = 32;
	
	for (j = start_inx; j < boundary && j < text_size; ++j) { 
	      D = (D<<1) | S[text[j]]; 
	      if (D < lim) {
		    k = 0;
		    h = j - pattern_size + 1;
		    while(k < p_len && pattern[k] == text[h+k]) k++;
		    if (k == p_len) match[j - pattern_size + 1] = 1; 
	      }
	} 
}

int preSo(unsigned char *pattern, int pattern_size, unsigned int *S) { 
   unsigned int j, lim; 
   int i; 
   for (i = 0; i < SIGMA; ++i) 
      S[i] = ~0; 
   for (lim = i = 0, j = 1; i < pattern_size; ++i, j <<= 1) { 
      S[pattern[i]] &= ~j; 
      lim |= j; 
   } 
   lim = ~(lim>>1); 
   return(lim); 
} 


