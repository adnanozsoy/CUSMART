#include "hash3.cuh"

__global__
void hash3(
        unsigned char *text, int text_size,
        unsigned char *pattern, int pattern_size,
        int *shift, int sh1, int stride_length, int *match){

         int sh, j;
	 unsigned char h = 0; 
	 unsigned long thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	 unsigned long start_inx = thread_id * stride_length;

	 unsigned long boundary = start_inx + stride_length + pattern_size;
	 boundary = boundary > text_size ? text_size : boundary;
	 unsigned long i;
	 if (pattern_size<3) return; 
	 
	 i = start_inx + (pattern_size - 1); 	  
	 while (i < boundary) {
	   sh1 = 1;
	   while (i < (boundary + 2) && sh != 0) {
	     h = text[i-2];
	     h = ((h<<1) + text[i-1]);
	     h = ((h<<1) + text[i]);
	     sh = shift[h];
	     i+=sh;
	   }

	   j=0; 
	   while(i < text_size && j<pattern_size && pattern[j]==text[i-(pattern_size-1)+j]) j++; 
	   if (j>=pattern_size) { 
	     match[i-(pattern_size - 1)] = 1; 
	   }
	   i+=sh1;
	 }
} 

