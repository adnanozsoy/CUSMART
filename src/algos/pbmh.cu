#include "pbmh.cuh"

__global__ void bmh_prob(unsigned char *text, unsigned long text_size,
			    unsigned char *pattern, int pattern_size,int *hbc, 
			    int *v, int search_len, int *match)
{
        int i;
	unsigned long thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned long start_inx = thread_id * search_len;
	unsigned long boundary = start_inx + search_len;
	boundary = boundary > text_size ? text_size : boundary;
	unsigned long s = start_inx;

	while (s < boundary && s <= text_size - pattern_size) {
	  i=0;
	  while(i<pattern_size && pattern[v[i]]==text[s+v[i]]) i++;
	  if (i==pattern_size) match[s] = 1;
	  s+=hbc[text[s+pattern_size-1]];
	}
}
