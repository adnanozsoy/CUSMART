#include "sfbom.cuh"

__global__
void sfbom(unsigned char *text, unsigned long text_size,
	   unsigned char *pattern, int pattern_size,int *LAMBDA,
	   int **trans, int search_len, int *match){
  
	unsigned long thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned long start_inx = thread_id * search_len;
	unsigned long boundary = start_inx + search_len + pattern_size;
	boundary = boundary > text_size ? text_size : boundary;
	int i, k, p;
	unsigned long j = start_inx + pattern_size;

	k=0;
	while(k < pattern_size && pattern[k] == text[k]) k++;
	if(k == pattern_size) match[k] = 1;
	//if ( !memcmp(x,y,m) ) count++;
	while (j < boundary && j < text_size) {
	  while (j > 0 && j < boundary && (p=FT(text[j+1],text[j]))>pattern_size ) j+=p-pattern_size; 
	  i = j-1; 
	  while (i >= 0 && p >= 0 && p < pattern_size+2 && (p = trans[p][text[i]]) != UNDEFINED ) i--; 
	  if (i < j-(pattern_size-1) && j<boundary && j > 0) { 
	    match[j] = 1; 
	    i++; 
	  } 
	  j = i + pattern_size;
	} 
}
