#include "tsw.cuh"

__global__
void two_sliding_window(unsigned char *text, unsigned long text_size,
                 unsigned char *pattern, int pattern_size, int **brBc_left,
                 int **brBc_right, int search_len, int *match) {
   
  	unsigned long thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned long start_inx = thread_id * search_len;

	unsigned long boundary = start_inx + search_len;
	boundary = boundary > text_size ? text_size : boundary;
	int i, a, b;
	unsigned long j = start_inx;
  
	/* Searching */
	a = text_size-pattern_size;
	while (j < boundary && j <= a) {
	  for (i=0; i<pattern_size && pattern[i]==text[j+i]; i++);
	  if (i>=pattern_size && j<=a) match[j] = 1;
	  
	  for (b=0; b<pattern_size && pattern[b]==text[a+b]; b++);
	  if (b>=pattern_size && j<a) match[j] = 1;
	  
	  j += brBc_left[text[j + pattern_size]][text[j + pattern_size + 1]];
	  a -= brBc_right[text[a - 1]][text[a - 2]];
	}
}

void preBrBcTSW(unsigned char *pattern, int pattern_size, int brBc[SIGMA][SIGMA]) {
   int a, b, i;
   for (a = 0; a < SIGMA; ++a)
      for (b = 0; b < SIGMA; ++b)
         brBc[a][b] = pattern_size + 2;
   for (a = 0; a < SIGMA; ++a)
      brBc[a][pattern[0]] = pattern_size + 1;
   for (i = 0; i < pattern_size - 1; ++i)
      brBc[pattern[i]][pattern[i + 1]] = pattern_size - i;
   for (a = 0; a < SIGMA; ++a)
      brBc[pattern[pattern_size - 1]][a] = 1;
}
