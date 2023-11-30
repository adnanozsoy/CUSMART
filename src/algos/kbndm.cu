#include "kbndm.cuh"

__global__
void factorized_backward_nondeterministic_dawg(
			unsigned char *text, int text_size, unsigned char *pattern, int pattern_size,
			int m1, unsigned int M, unsigned int **B, unsigned int *L,
			int stride_length, int *match){
  
      	unsigned long thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned long start_inx = thread_id * stride_length;
	unsigned long boundary = start_inx + stride_length + m1 - 1;
	boundary = boundary > text_size ? text_size : boundary;
	int i;
	unsigned long j = start_inx + m1 - 1;

	i = pattern_size-m1;
	while (j < boundary && j < text_size) {
	  int k = 1;
	  int l = text[j] == pattern[0];
	  unsigned char c = text[j];
	  unsigned int D = ~0;
	  unsigned int D_;
	  do {
	    D = D & B[c][text[j-k]];
	    D_ = D & L[c];
	    D += D_;
	    c = text[j-k];
	    k++;
	    if (D & M) {
	      if (k == m1) {
		while (i>0  && *pattern+m1 && (*pattern+m1 == *text+j+1)){
		  ++pattern;
		  ++text;
		  --i;
		}
		if (i == 0){
		  match[j] = 1;
		}
		/*if (!strncmp(pattern+m1, text+j+1, pattern_size-m1)) {
                  match[j] = 1;
		  }*/
		break;
	      }
	      l = k;
	    }
	  } while (D);
	  j += m1-l;
	}
}
