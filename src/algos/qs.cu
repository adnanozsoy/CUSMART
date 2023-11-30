
#include "qs.cuh"


__host__
void preQsBc(unsigned char *pattern, int pattern_size, int *qbc) {
   for(int i = 0; i < SIGMA; ++i)
      qbc[i] = pattern_size + 1;
   for(int i = 0; i < pattern_size; ++i)
      qbc[pattern[i]] = pattern_size - i;
}

__global__
void quicksearch( unsigned char *text, int text_size,
                  unsigned char *pattern, int pattern_size,
                  int *bmBc, int stride_length, int *match) {
   int i;

   unsigned long idx = (threadIdx.x + blockIdx.x*blockDim.x) * stride_length;
   if ( idx > text_size - pattern_size ) return;

   unsigned long upper_limit;
   if (idx <= text_size - pattern_size - stride_length)
      upper_limit = stride_length + idx;
   else
      upper_limit = text_size - pattern_size;
   
   /* Searching */
   unsigned long s = idx;
   while(s <= upper_limit) {
      i=0;
      while(i < pattern_size && pattern[i] == text[s + i]) i++;
      if(i == pattern_size) match[s] = 1;
      s += bmBc[ text[s + pattern_size] ];
   }
}