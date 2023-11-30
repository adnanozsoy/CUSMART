
#include "ssabs.cuh"

__host__
void preQsBcSSABS(unsigned char *pattern, int pattern_size, int *qbc){
  int i; 
  for (i=0;i<SIGMA;i++)	qbc[i]=pattern_size+1; 
  for (i=0;i<pattern_size;i++) qbc[pattern[i]]=pattern_size-i; 
}

__global__
void ssabs( 
        unsigned char *text, int text_size,
        unsigned char *pattern, int pattern_size,
        int *qsBc, unsigned char firstCh, unsigned char lastCh, int stride_length, int *match)
{
	unsigned long thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned long start_inx = thread_id * stride_length;
	
	unsigned long boundary = start_inx + stride_length;
	boundary = boundary > text_size ? text_size : boundary;
	int i;
	unsigned long j = start_inx;

	while(j < boundary && j <= text_size - pattern_size){ 
	  // Stage 1 
	  if (lastCh == text[j + pattern_size - 1] && firstCh == text[j]) 
	    { 
	      //Stage 2 
	      for (i = pattern_size-2; i > 0 && pattern[i] == text[j + i]; i--); 
	      if (i <= 0) match[j] = 1; 
	    } 
	  // Stage 3 
	  j += qsBc[text[j + pattern_size]]; 
	} 
}

