#include "bsdm.cuh"

__global__
void backward_snr_dawg_matching(
                                unsigned char *text, int text_size,
                                unsigned char *pattern, int pattern_size, unsigned int *B,
                                unsigned int *pos, int len, int start, int search_len, int *match){

        int i, k, p;
  	unsigned long thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned long start_inx = thread_id * search_len;
	unsigned long boundary = start_inx + search_len + len - 1;
	boundary = boundary > text_size ? text_size : boundary;
       
	unsigned long j = start_inx + len - 1;
	unsigned char *xstart = pattern+start;

	int offset = len+start-1;
	while(j<boundary && j<text_size) {
	  while ((i=pos[text[j]])<0) j+=len;
	  k=1;
	  while(k<=i && xstart[i-k]==text[j-k]) k++;
	  if (k>i) {
	    if (k==len) {
	      p = 0;
	      while(p < pattern_size && pattern[p] == text[p+j-offset]) p++;
	      if(p == pattern_size) if (j-offset<=text_size-pattern_size) match[j] = 1;
	      //if (!memcmp(pattern,text+j-offset,pattern_size)) if (j-offset<=text_size-pattern_size) match[i] = 1;
	    }
	    else j-=k;
	  }
	  j+=len;
	}
}

