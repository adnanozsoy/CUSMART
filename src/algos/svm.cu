/*
 * SMART: string matching algorithms research tool.
 * Copyright (C) 2012  Simone Faro and Thierry Lecroq
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>
 * 
 * contact the authors at: faro@dmi.unict.it, thierry.lecroq@univ-rouen.fr
 * download the tool at: http://www.dmi.unict.it/~faro/smart/
 *
 * This is an implementation of the Shift Vector Matching algorithm
 * in H. Peltola and J. Tarhio. 
 * Alternative Algorithms for Bit-Parallel String Matching. 
 * Proceedings of the 10th International Symposium on String Processing and Information Retrieval SPIRE'03, (2003).
 */

#include "svm.cuh"


__global__
void svm(unsigned char *text, int text_size, 
        unsigned char *pattern, int pattern_size,
        unsigned int *cv, int stride_length, int *match)
{
   int upper_limit;
   int idx = (threadIdx.x + blockIdx.x*blockDim.x) * stride_length;
   if ( idx > text_size - pattern_size ) return;

   if (idx <= text_size - pattern_size - stride_length)
       upper_limit = idx + pattern_size + stride_length;
   else
       upper_limit = text_size;

   /* searching */
   if (idx == 0)
   {
      int firstcheck = 1;
      for (int i = 0; i < pattern_size; ++i)
         if (text[idx+i] != pattern[idx+i]) {firstcheck = 0; break;}
      if (firstcheck) match[0] = 1;
   }
   
   int sv = 0;   
   int s = idx + pattern_size;
   while(s < upper_limit){
      sv |= cv[text[s]];
      int j = 1;
      while((sv & 1) == 0) {
         sv |= (cv[text[s-j]] >> j);
         if(j >= pattern_size) {match[s] = 1; break;}
         ++j;
      }
      sv >>= 1; s += 1;
      while((sv & 1)==1) {   
         sv >>= 1;
         s += 1;
      }
   }
}

/*
 * Shift Vector Matching algorithm designed for large patterns
 * The present implementation searches for prefixes of the pattern of length 32.
 * When an occurrence is found the algorithm tests for the whole occurrence of the pattern
 */

__global__
void svm_large(unsigned char *text, int text_size, 
        unsigned char *pattern, int pattern_size,
        unsigned int *cv, int stride_length, int *match) 
{
   int upper_limit;
   int idx = (threadIdx.x + blockIdx.x*blockDim.x) * stride_length;
   if ( idx > text_size - pattern_size ) return;

   if (idx <= text_size - pattern_size - stride_length)
       upper_limit = stride_length + idx;
   else
       upper_limit = text_size;

   int p_len = pattern_size;
   pattern_size = 32;

   /* Searching */
   
   char firstcheck = 1;
   for (int i = 0; i < pattern_size; ++i)
      if (text[idx+i] != pattern[idx+i]) {firstcheck = 0; break;}
   if (firstcheck) match[0] = 1;

   int sv = 0;   
   int s = idx + pattern_size;
   while(s < upper_limit){
      sv |= cv[text[s]];
      int j = 1;
      while((sv & 1) == 0) {
         sv |= (cv[text[s-j]] >> j);
         if(j >= pattern_size) {
            int k = pattern_size; 
            int first = s-pattern_size+1;
            while (k<p_len && pattern[k]==text[first+k]) k++;
            if (k==p_len) match[first] = 1; 
            break;
         }
         j++;
      }
      sv >>= 1; s += 1;
      while((sv & 1)==1) {   
         sv >>= 1;
         s += 1;
      }
   }
}

