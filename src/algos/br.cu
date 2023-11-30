#include "br.cuh"

__global__ 
void berry_ravindran(unsigned char *text, unsigned long text_size,
	unsigned char *pattern, int pattern_size,int *brBc, 
	int search_len, int *match)
{
	unsigned long thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned long start_inx = thread_id * search_len;
	
	unsigned long boundary = start_inx + search_len + pattern_size - 1;
	boundary = boundary > text_size ? text_size : boundary;
	int i;
	unsigned long j = start_inx;

	while (j < boundary && j <= text_size - pattern_size) {
		for (i = 0; i < pattern_size && pattern[i] == text[i + j]; i++);
		if (i >= pattern_size) {
			match[j] = 1;
		}
		
		j += getBrBc(text[j + pattern_size], text[j + pattern_size + 1]);
	}
}


void preBrBc(unsigned char *x, int m, int *brBc) { 
   int a, b, i; 
   for (a = 0; a < SIGMA; ++a) 
      for (b = 0; b < SIGMA; ++b) 
         getBrBc(a,b) = m + 2; 
   for (a = 0; a < SIGMA; ++a) 
      getBrBc(a, x[0]) = m + 1; 
   for (i = 0; i < m - 1; ++i) 
      getBrBc(x[i], x[i + 1]) = m - i; 
   for (a = 0; a < SIGMA; ++a) 
      getBrBc(x[m - 1], a) = 1; 
} 
