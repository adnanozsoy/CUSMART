#include "mp.cuh"

__host__
void pre_morris_pratt(unsigned char *pattern, int pattern_size, int *shift_array) {
   int i, j;
   i = 0;
   j = shift_array[0] = -1;
   while (i < pattern_size) {
      while (j > -1 && pattern[i] != pattern[j])
         j = shift_array[j];
      i++;
      j++;
      shift_array[i] = j;
   }
}

__global__
void morris_pratt(unsigned char *text, unsigned long text_size, 
                  unsigned char *pattern, int pattern_size, 
                  int *shift_array, int stride_length, int *match){

   unsigned long idx = (threadIdx.x + blockIdx.x*blockDim.x) * stride_length;
   if ( idx > text_size - pattern_size ) return;

   int i = 0;
   unsigned long j = idx;

   while( (j-i) < idx + stride_length) {

      while(i > -1 && text[j] != pattern[i])
         i = shift_array[i];

      i++;
      j++;

      if (i >= pattern_size){
         match[j-i] = 1;
         i = shift_array[i];
      }

   }
}