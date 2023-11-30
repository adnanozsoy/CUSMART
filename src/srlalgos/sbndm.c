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
 * This is an implementation of the Simplified Backward Nondeterministic DAWG Matching algorithm
 * in H. Peltola and J. Tarhio.
 * Alternative Algorithms for Bit-Parallel String Matching. 
 * Proceedings of the 10th International Symposium on String Processing and Information Retrieval SPIRE'03, Lecture Notes in Computer Science, vol.2857, pp.80--94, Springer-Verlag, Berlin, Manaus, Brazil, (2003).
 */

#include "sbndm.h"
#include "include/define.h"

#include "stdio.h"
#include "stdlib.h"
#include "string.h"

static void sbndm_large(search_parameters params);

void sbndm(search_parameters params) {
   int j,i, last, first;
   unsigned int D, B[SIGMA], s;
   int mM1 = params.pattern_size-1;
   int mM2 = params.pattern_size-2;
   int count = 0, restore[XSIZE+1], shift;

   if (params.pattern_size>32) return sbndm_large(params);  

   /* Preprocessing */
   for (i=0; i<SIGMA; i++)  B[i] = 0;
   for (i=0; i<params.pattern_size; i++) B[params.pattern[params.pattern_size-i-1]] |= (unsigned int)1 << (i+WORD-params.pattern_size);

   last = params.pattern_size;
   s = (unsigned int)(~0) << (WORD-params.pattern_size);
   s = (unsigned int)(~0);
   for (i=params.pattern_size-1; i>=0; i--) {
      s &= B[params.pattern[i]]; 
      if (s & ((unsigned int)1<<(WORD-1))) {
         if (i > 0)  last = i; 
      }
      restore[i] = last;
      s <<= 1;
   }
        shift = restore[0];

   for (i=0; i<params.pattern_size; i++) params.text[params.text_size+i]=params.pattern[i];

   /* Searching */
   if (!memcmp(params.pattern, params.text, params.pattern_size)) params.match[0] = 1;
   i = params.pattern_size;
   while (1) {
      D = B[params.text[i]];
      j = i-1; first = i-params.pattern_size+1;
      while (1) {
         D = (D << 1) & B[params.text[j]];
         if (!((j-first) && D)) break;
         j--;
      }
      if (D != 0) {
         if (i >= params.text_size) return;
         params.match[first] = 1; //OUTPUT(first);
         i += shift;
      } 
      else {
         i = j+params.pattern_size;
      }
   }
}

/*
 * Simplified Backward Nondeterministic DAWG Matching algorithm designed for large patterns
 * The present implementation searches for prefixes of the pattern of length 32.
 * When an occurrence is found the algorithm tests for the whole occurrence of the pattern
 */

static void sbndm_large(search_parameters params) {
   int j,i, last, first, p_len, k;
   unsigned int D, B[SIGMA], s;
   int mM1 = params.pattern_size-1;
   int mM2 = params.pattern_size-2;
   int restore[XSIZE+1], shift;

   p_len = params.pattern_size;
   params.pattern_size = 32;
   int diff = p_len-params.pattern_size;

   /* Preprocessing */
   for (i=0; i<SIGMA; i++)  B[i] = 0;
   for (i=0; i<params.pattern_size; i++) B[params.pattern[params.pattern_size-i-1]] |= (unsigned int)1 << (i+WORD-params.pattern_size);

   last = params.pattern_size;
   s = (unsigned int)(~0) << (WORD-params.pattern_size);
   s = (unsigned int)(~0);
   for (i=params.pattern_size-1; i>=0; i--) {
      s &= B[params.pattern[i]]; 
      if (s & ((unsigned int)1<<(WORD-1))) {
         if (i > 0)  last = i; 
      }
      restore[i] = last;
      s <<= 1;
   }
   shift = restore[0];

   for (i=0; i<params.pattern_size; i++) params.text[params.text_size+i]=params.pattern[i];

   /* Searching */
   if (!memcmp(params.pattern, params.text, params.pattern_size)) params.match[0] = 1;
   i = params.pattern_size;
   while (1) {
      while ((D = B[params.text[i]]) == 0) i += params.pattern_size;
      j = i-1; first = i-params.pattern_size+1;
      while (1) {
         D = (D << 1) & B[params.text[j]];
         if (!((j-first) && D)) break;
         j--;
      }
      if (D != 0) {
         if (i+diff >= params.text_size) return;
         k=params.pattern_size;
         while(k<params.pattern_size && params.pattern[k]==params.text[first+k]) k++;
         if (k==params.pattern_size) params.match[first] = 1; //OUTPUT(first);
         i += shift;
      } 
      else {
         i = j+params.pattern_size;
      }
   }
}
