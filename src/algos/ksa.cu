#include "kbndm.cuh"

__global__
void factorized_shift_and(
			unsigned char *text, int text_size, unsigned char *pattern, int pattern_size,
			int m1, unsigned int M, unsigned int **B, unsigned int *L,
			int stride_length, int *match){
  
      	unsigned long thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned long start_inx = thread_id * stride_length;
	unsigned long boundary = start_inx + stride_length + pattern_size;
	boundary = boundary > text_size ? text_size : boundary;
	int i;
	unsigned long j;
	unsigned int D, D_;
	unsigned char c;
	
	D = 0;
	c = text[0];
	
	i = pattern_size-m1;
        for (j = start_inx + 1; j < boundary && j < text_size; j++) {
	  D = (D|1) & B[c][text[j]];
	  D_ = D & L[c];
	  D += D_;
	  c = text[j];
	  if (D & M) {
	    while (i>0  && *pattern+m1 && (*pattern+m1 == *text+j+1)){
	      ++pattern;
	      ++text;
	      --i;
	    }
	    if (i == 0){
	      match[j] = 1;
	    }
	    //if (!strncmp(x+m1, y+j+1, m-m1)) {
	    // match[j] = 1;
	  }
	}
}
