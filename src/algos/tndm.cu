#include "tndm.cuh"


__global__
void two_way_nondeterministic_dawg(
   unsigned char *text, int text_size, 
   unsigned char *pattern, int pattern_size,
   unsigned int *B, unsigned int *restore, int stride_length, int *match)
{
   int upper_limit;
   int idx = (threadIdx.x + blockIdx.x*blockDim.x) * stride_length;
   if ( idx > text_size - pattern_size ) return;

   if (idx <= text_size - pattern_size - stride_length)
      upper_limit = idx + stride_length + pattern_size;
   else
      upper_limit = text_size;


   /* Searching */
   int j = idx + pattern_size - 1;
   while (j <= upper_limit){
      int i = 0;
      int last = pattern_size;
      unsigned int d = B[text[j]];
      if ((d&1) == 0) {
         while (d!=0 && !(d&((unsigned int)1<<i))) {
            i++;
            d &= B[text[j+i]]<<i;
         } 
         if (d==0 || j+i>=text_size ) {
            j += last; 
            continue;
         } 
         else {
            j += i; 
            last = restore[i]; 
         }
      }
      do {
         i++;
         if (d & ((unsigned int)1<<(pattern_size-1))) {
            if(i < pattern_size)  last = pattern_size-i; 
            else {
               match[j] = 1; 
               break;
            } 
         }
         d<<=1;
         d &= B[text[j-i]]; 
      } while(d != 0); 

      j += last; 
   } 
}

/*
 * Two-Way Nondeterministic DAWG Matching algorithm designed for large patterns
 * The present implementation searches for prefixes of the pattern of length 32.
 * When an occurrence is found the algorithm tests for the whole occurrence of the pattern
 */

__global__
void two_way_nondeterministic_dawg_large(
   unsigned char *text, int text_size, 
   unsigned char *pattern, int pattern_size,
   unsigned int *B, unsigned int *restore, int stride_length, int *match)
{
   int upper_limit;
   int idx = (threadIdx.x + blockIdx.x*blockDim.x) * stride_length;
   if ( idx > text_size - pattern_size ) return;

   if (idx <= text_size - stride_length)
      upper_limit = idx + stride_length;
   else
      upper_limit = text_size;


   /* Searching */
   int pat_len = 32;
   int j = idx + pat_len - 1;
   while (j <= upper_limit){
      int i = 0;
      int last = pat_len;
      unsigned int d = B[text[j]];
      if ((d&1) == 0) {
         while (d!=0 && !(d&((unsigned int)1<<i))) {
            i++;
            d &= B[text[j+i]]<<i;
         } 
         if (d==0 || j+i>=text_size ) {
            j += last; 
            continue;
         } 
         j += i; 
         last = restore[i]; 
      }
      do {
         i++;
         if (d & ((unsigned int)1<<(pat_len-1))) {
            if(i < pat_len)  last = pat_len-i; 
            else {
               int k = pat_len;
               while(k<pattern_size && pattern[k]==text[j-pat_len+1+k]) k++;
               if (k>=pattern_size) match[j-pat_len+1] = 1;
               break;
            } 
         }
         d<<=1;
         d &= B[text[j-i]]; 
      } while(d != 0); 

      j += last; 
   } 
}
