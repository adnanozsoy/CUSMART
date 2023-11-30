
#include "hor.cuh"

__global__ void horspool(unsigned char *text, unsigned long text_size, unsigned char *pattern,
	   	int pattern_size, unsigned char hbc[], int stride_length, int *match) {

           int i; 

	 unsigned long thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	 unsigned long start_inx = thread_id * stride_length;

	 unsigned long boundary = start_inx + stride_length;
	 boundary = boundary > text_size ? text_size : boundary;
	 unsigned long j = start_inx;

  /*
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

        int start_index = tid * stride_length;
        int end_index = (tid + 1) * stride_length + pattern_size - 1;
  */
	//        match[tid] = 0;

        while(j < boundary && j <= text_size - pattern_size) {
		i = 0;
		while(i < pattern_size && pattern[i] == text[j + i]) i++; 
		if(i == pattern_size) match[j] = 1; 
		j += hbc[text[j + pattern_size - 1]];
	}
}										              
												      

void pre_horspool(unsigned char *pattern, int pattern_size, unsigned char hbc[]){
     int i;
     for(i = 0; i < SIGMA; i++) hbc[i] = pattern_size;
     for(i = 0; i < pattern_size - 1; i++) hbc[pattern[i]] = pattern_size - i - 1;
}
