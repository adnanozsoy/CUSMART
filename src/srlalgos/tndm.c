/*
 * CUSMART: CUDA string matching algorithms research tool.
 * Copyright (C) 2019  CUSMART
 * Based on SMART project Copyright (C) 2012 Simone Faro and Thierry Lecroq
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
 * This is an implementation of the Two-Way Nondeterministic DAWG Matching algorithm
 * in H. Peltola and J. Tarhio.
 * Alternative Algorithms for Bit-Parallel String Matching. 
 * Proceedings of the 10th International Symposium on String Processing and Information Retrieval SPIRE'03, Lecture Notes in Computer Science, vol.2857, pp.80--94, Springer-Verlag, Berlin, Manaus, Brazil, (2003).
 */

#include "tndm.h"
#include "include/define.h"

static void tndm_large(search_parameters params);

void tndm(search_parameters params) {
   int i,j,s,last, restore[XSIZE+1];
   unsigned int d, B[SIGMA];
   int count = 0;

   if(params.pattern_size>32){
      tndm_large(params);
      return;
   }

   /* Preprocessing */
   for(i=0; i<SIGMA; i++) B[i]=0;
   s=1; 
   for (i=params.pattern_size-1; i>=0; i--){ 
      B[params.pattern[i]] |= s; 
      s <<= 1;
   }
   last = params.pattern_size;
   s = (unsigned int)(~0) >> (WORD-params.pattern_size);
   for (i=params.pattern_size-1; i>=0; i--) {
      s &= B[params.pattern[i]]; 
      if (s & ((unsigned int)1<<(params.pattern_size-1))) {
         if(i > 0)  last = i; 
      }
      restore[params.pattern_size-i] = last;
      s <<= 1;
   }

   /* Searching */
   j=params.pattern_size-1;
   while (j < params.text_size){
      i=0;
      last=params.pattern_size;
      d = B[params.text[j]];
      if ((d&1) == 0) {
         while (d!=0 && !(d&((unsigned int)1<<i))) {
            i++;
            d &= B[params.text[j+i]]<<i;
         } 
         if (d==0 || j+i>=params.text_size ) {
            goto over;
         } 
         j += i; 
         last = restore[i]; 
      }
      do {
         i++;
         if (d & ((unsigned int)1<<(params.pattern_size-1))) {
            if(i < params.pattern_size)  last = params.pattern_size-i; 
            else {
               params.match[j] = 1; 
               goto over;
            } 
         }
         d<<=1;
         d &= B[params.text[j-i]]; 
      } while(d != 0); 

      over:
      j += last; 
   } 
}

/*
 * Two-Way Nondeterministic DAWG Matching algorithm designed for large patterns
 * The present implementation searches for prefixes of the pattern of length 32.
 * When an occurrence is found the algorithm tests for the whole occurrence of the pattern
 */

static void tndm_large(search_parameters params)
{
   int i,j,s,last, restore[XSIZE+1], p_len, k;
   unsigned int D, B[SIGMA];
   int count = 0;
   p_len = params.pattern_size;
   params.pattern_size = 32;

   /* Preprocessing */
   for(i=0; i<SIGMA; i++) B[i]=0;
   s=1; 
   for (i=params.pattern_size-1; i>=0; i--){ 
      B[params.pattern[i]] |= s; 
      s <<= 1;
   }
   last = params.pattern_size;
   s = (unsigned int)(~0) >> (WORD-params.pattern_size);
   for (i=params.pattern_size-1; i>=0; i--) {
      s &= B[params.pattern[i]]; 
      if (s & ((unsigned int)1<<(params.pattern_size-1))) {
         if(i > 0)  last = i; 
      }
      restore[params.pattern_size-i] = last;
      s <<= 1;
   }

   /* Searching */
   j=params.pattern_size-1;
   while (j < params.text_size){
      i=0;
      last=params.pattern_size;
      D = B[params.text[j]];
      if ((D & 1) == 0) {
         while (D!=0 && !(D & ((unsigned int)1<<i))) {
            i++;
            D &= B[params.text[j+i]]<<i;
         } 
         if (D==0 || j+i>=params.text_size ) {
            goto over;
         } 
         j += i; 
         last = restore[i]; 
      }

      do {
         i++;
         if (D & ((unsigned int)1<<(params.pattern_size-1))) {
            if(i < params.pattern_size)  last = params.pattern_size-i; 
            else {
               k = params.pattern_size;
               while(k<p_len && params.pattern[k]==params.text[j-params.pattern_size+1+k]) k++;
               if (k>=p_len) params.match[j-params.pattern_size+1] = 1;
               goto over;
            } 
         }
         D <<= 1;
         D &= B[params.text[j-i]]; 
      } while(D != 0); 

      over:
      j += last; 
   } 
}
