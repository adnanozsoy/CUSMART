#include "raita.cuh"
#include <stdio.h>
#include <string.h>

__global__ void raita(unsigned char *text, unsigned long text_size, unsigned char *pattern, int pattern_size,
		      unsigned char bmBc[], char firstCh, char middleCh,
		      char lastCh, int stride_length, int *match){
        int i;
	unsigned long thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned long start_inx = thread_id * stride_length;
	
	unsigned long boundary = start_inx + stride_length;
	boundary = boundary > text_size ? text_size : boundary;
	unsigned long j = start_inx;
	
	while (j < boundary && j <= text_size - pattern_size) {
	      i = 0;
	      if (lastCh == text[j + pattern_size - 1] &&
		  middleCh == text[j + pattern_size/2] && firstCh == text[j]){
		
		    if(pattern_size == 1 || pattern_size == 2) match[j] = 1;
		    else {
		          while(i < pattern_size - 2 && pattern[i + 1] == text[j + i + 1]) i++;
			  if(i == pattern_size - 2) match[j] = 1;
		    }
	      }
	      j += bmBc[text[j + pattern_size - 1]];
	}
}

void preBmBcRAITA(unsigned char *pattern, int pattern_size, unsigned char bmBc[]) {
  int i;

  for (i = 0; i < SIGMA; ++i)
    bmBc[i] = pattern_size;
  for (i = 0; i < pattern_size - 1; ++i)
    bmBc[pattern[i]] = pattern_size - i - 1;
}
