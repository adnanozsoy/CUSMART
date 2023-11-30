
#include "fsbndm.cuh"

/*
 * Forward Semplified BNDM algorithm designed for large patterns
 * The present implementation searches for prefixes of the pattern of length 32.
 * When an occurrence is found the algorithm tests for the whole occurrence of the pattern
 */
__global__
void forward_simplified_backward_nondeterministic_dawg_matching_large(
   unsigned char *text, unsigned long text_size,
   unsigned char *pattern, int pattern_size, 
   unsigned int *B, int stride_length, int *match)
{
    unsigned int upper_limit;
    int idx = (threadIdx.x + blockIdx.x*blockDim.x) * stride_length;
    if ( idx > text_size - pattern_size ) return;

    if (idx <= text_size - pattern_size - stride_length)
        upper_limit = idx + stride_length + pattern_size;
    else
        upper_limit = text_size;

   int p_len = 32;

   /* Searching */
   //if(!memcmp(params.pattern,params.text,p_len)) params.match[0] = 1;
   int j = idx + p_len;
   while (j < upper_limit) {
      unsigned int D = (B[text[j+1]]<<1) & B[text[j]];
      if (D != 0) {
         int pos = j;
         while (D=(D<<1) & B[text[j-1]]) --j;
         j += p_len - 1;
         if (j == pos) {
            int k = p_len; 
            int s = j - p_len - 1;
            while (k<pattern_size && pattern[k]==text[s+k]) k++;
            if (k==pattern_size) match[s] = 1;
            ++j;
         }
      }
      else j+=pattern_size;
   }
}

__global__
void forward_simplified_backward_nondeterministic_dawg_matching(
   unsigned char *text, unsigned long text_size,
   unsigned char *pattern, int pattern_size, 
   unsigned int *B, int stride_length, int *match)
{
    unsigned int upper_limit;
    int idx = (threadIdx.x + blockIdx.x*blockDim.x) * stride_length;
    if ( idx > text_size - pattern_size ) return;

    if (idx <= text_size - pattern_size - stride_length)
        upper_limit = idx + stride_length + pattern_size;
    else
        upper_limit = text_size;
   
   /* Searching */
   //if(!memcmp(params.pattern,params.text,params.pattern_size)) params.match[0] = 1;
   int j = idx + pattern_size;
   while (j < upper_limit) {
      unsigned int D = (B[text[j+1]]<<1) & B[text[j]];
      if (D != 0) {
         int pos = j;
         while (D=(D<<1) & B[text[j-1]]) --j;
         j += pattern_size - 1;
         if (j == pos) {
            match[j] = 1;
            ++j;
         }
      }
      else j+=pattern_size;
   }
}
