#include "kmpskip.cuh"

__global__
void kmpskip(unsigned char *text, unsigned long text_size, unsigned char *pattern,
	   int pattern_size, int *kmpNext, int *list, int *mpNext,
	   int *z, int search_len, int *match) {

        int i, k, kmpStart, per, start, wall;

	unsigned long thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned long start_inx = thread_id * search_len;
	
	unsigned long boundary = start_inx + search_len;
	boundary = boundary > text_size ? text_size : boundary;
	unsigned long j;
	
	wall = 0;
	per = pattern_size - kmpNext[pattern_size];
	j = start_inx - 1;
	i = -1;
	do {
	  j += pattern_size;
	} while (j < boundary && j < text_size && z[text[j]] < 0);
	if (j >= text_size)
	  return;
	i = z[text[j]];
	start = j - i;
	while (start < boundary && start <= text_size - pattern_size) {
	  if (start > wall)
	    wall = start;
	  k = attempt(text, pattern, pattern_size, start, wall);
	  wall = start + k;
	  if (k == pattern_size) {
	    match[start] = 1;
	    i -= per;
	  }
	  else
	    i = list[i];
	  if (i < 0) {
	    do {
	      j += pattern_size;
	    } while (j < boundary && j < text_size && z[text[j]] < 0);
	    if (j >= text_size)
	      return;
	    i = z[text[j]];
	  }
	  kmpStart = start + k - kmpNext[k];
	  k = kmpNext[k];
	  start = j - i;
	  while (start < kmpStart ||
		 (kmpStart < start && start < wall)) {
	    if (start < kmpStart) {
	      i = list[i];
	      if (i < 0) {
		do {
                  j += pattern_size;
		} while (j < boundary && j < text_size && z[text[j]] < 0);
		if (j >= text_size)
                  return;
		i = z[text[j]];
	      }
	      start = j - i;
	    }
	    else {
	      kmpStart += (k - mpNext[k]);
	      k = mpNext[k];
	    }
	  }
	}
}

__host__
void preKmp(unsigned char *pattern, int pattern_size, int kmpNext[]) {
   int i, j;
   i = 0;
   j = kmpNext[0] = -1;
   while (i < pattern_size) {
      while (j > -1 && pattern[i] != pattern[j])
         j = kmpNext[j];
      i++;
      j++;
      if (i<pattern_size && pattern[i] == pattern[j])
         kmpNext[i] = kmpNext[j];
      else
         kmpNext[i] = j;
   }
}

__host__
void preMp(unsigned char *pattern, int pattern_size, int mpNext[]) {
   int i, j;

   i = 0;
   j = mpNext[0] = -1;
   while (i < pattern_size) {
      while (j > -1 && pattern[i] != pattern[j])
         j = mpNext[j];
      mpNext[++i] = ++j;
   }
}

__host__ __device__
int attempt(unsigned char *text, unsigned char *pattern, int pattern_size, int start, int wall) {
   int k;
   k = wall - start;
   while (k < pattern_size && pattern[k] == text[k + start])
      ++k;
   return(k);
}
