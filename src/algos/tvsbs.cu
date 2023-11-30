#include "tvsbs.cuh"

__global__
void tvsbs(unsigned char *text, unsigned long text_size,
	   unsigned char *pattern, int pattern_size, int **BrBc,
	   char firstCh, char lastCh, int search_len, int *match){

	unsigned long thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned long start_inx = thread_id * search_len;

	unsigned long boundary = start_inx + search_len;
	boundary = boundary > text_size ? text_size : boundary;
	int i;
	unsigned long j = start_inx;

	while(j < boundary && j <= text_size - pattern_size){
	      if (lastCh == text[j + pattern_size - 1] && firstCh == text[j]) {
		    for (i = pattern_size-2; i > 0 && pattern[i] == text[j + i]; i--);
		        if (i <= 0) match[j] = 1;
	      }
	      j += BrBc[text[j + pattern_size]][text[j+pattern_size+1]];
	}
}

void TVSBSpreBrBc(unsigned char *pattern, int pattern_size, int brBc[SIGMA][SIGMA]) {
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
