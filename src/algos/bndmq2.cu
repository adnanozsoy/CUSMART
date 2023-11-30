
#include "bndmq2.cuh"

#include <stdlib.h>
#include <string.h>

__global__
void backward_nondeterministic_dawg_qgram(
   unsigned char *text, int text_size, 
   unsigned char *pattern, int pattern_size,
   unsigned int *B, unsigned int M, int stride_length, int *match)
{
   int q = 2;
    int upper_limit;
    int idx = (threadIdx.x + blockIdx.x * blockDim.x) * stride_length;
    if ( idx > text_size - pattern_size ) return;
    if (idx <= text_size - pattern_size - stride_length)
        upper_limit = idx + pattern_size + stride_length;
    else
        upper_limit = text_size;

   /* Searching */
   // if(!memcmp(pattern,text,pattern_size)) match[0] = 1;
   int i = idx + pattern_size+1-q;
   while (i <= upper_limit - q) {
      unsigned int D = (B[text[i+1]]<<1)&B[text[i]]; // GRAM2
      if (D != 0) {
         int j = i;
         int first = i - (pattern_size - q);
         do {
            if ( D >= M ) {
               if (j > first) i = j-1;
               else match[first] = 1;
            }
            j = j-1;
            D = (D<<1) & B[text[j]];
         } while (D != 0);
      }
      i = i+pattern_size-q+1;
   }
}


__global__
void backward_nondeterministic_dawg_qgram_large(
   unsigned char *text, int text_size, 
   unsigned char *pattern, int pattern_size,
   unsigned int *B, unsigned int M, int stride_length, int *match)
{
   int q = 2;
    int upper_limit;
    int idx = (threadIdx.x + blockIdx.x * blockDim.x) * stride_length;
    if ( idx > text_size - pattern_size ) return;
    if (idx <= text_size - pattern_size - stride_length)
        upper_limit = idx + pattern_size + stride_length;
    else
        upper_limit = text_size;

   int p_len = 32;

   /* Searching */
   //if(!memcmp(params.pattern,params.text,params.pattern_size)) params.match[0] = 1;
   int i = idx + p_len+1-q;
   while (i <= upper_limit - q) {
      unsigned int D = (B[text[i+1]]<<1)&B[text[i]]; // GRAM2
      if (D != 0) {
         int j = i;
         int first = i - (p_len - q);
         do {
            if ( D >= M ) {
               if (j > first) i = j-1;
               else {
                  int k = p_len;
                  while(k<pattern_size && pattern[k]==text[first+k]) k++;
                  if(k==pattern_size) match[first] = 1;
               }
            }
            j = j-1;
            D = (D<<1) & B[text[j]];
         } while (D != 0);
      }
      i = i+p_len-q+1;
   }
}


