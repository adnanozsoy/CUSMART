#include "smith.cuh"

__global__ void smith(unsigned char *text, unsigned long text_size, unsigned char *pattern, int pattern_size,
		      unsigned char bmBc[], unsigned char qsBc[], int stride_length, int *match){
         int i; 

	 unsigned long thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	 unsigned long start_inx = thread_id * stride_length;

	 unsigned long boundary = start_inx + stride_length;
	 boundary = boundary > text_size ? text_size : boundary;
	 unsigned long j = start_inx;
	 	 
      	 while (j < boundary && j <= text_size - pattern_size) {
	        i = 0;
		while(i < pattern_size && pattern[i] == text[j + i]) i++;
		 
		if(i == pattern_size) match[j] = 1;
      		j += MAX(bmBc[text[j + pattern_size - 1]], qsBc[text[j + pattern_size]]);
	 }
}

void preQsBcSMITH(unsigned char *pattern, int pattern_size, unsigned char qsBc[])
{
  int i;
  for (i = 0; i < SIGMA; i++) qsBc[i] = pattern_size + 1;
  for (i = 0; i < pattern_size; i++) qsBc[pattern[i]] = pattern_size - i;
}

void preBmBcSMITH(unsigned char *pattern, int pattern_size, unsigned char bmBc[]) {
  int i;

  for (i = 0; i < SIGMA; ++i)
    bmBc[i] = pattern_size;
  for (i = 0; i < pattern_size - 1; ++i)
    bmBc[pattern[i]] = pattern_size - i - 1;
}
