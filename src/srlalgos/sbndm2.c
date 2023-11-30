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
 * This is an implementation of the Simplified BNDM with loop unrolling algorithm
 * in J. Holub and B. Durian.
 * Talk: Fast variants of bit parallel approach to suffix automata. The Second Haifa Annual International Stringology Research Workshop of the Israeli Science Foundation, (2005).
 */

#include "sbndm2.h"
#include "include/define.h"

#include <stdlib.h>
#include <string.h>

static void sbndm2_large(search_parameters params);

void sbndm2(search_parameters params) 
{
   unsigned int B[SIGMA], D;
   int i, j, pos, mMinus1, m2, shift;
   if(params.pattern_size>32) {
      sbndm2_large(params);
      return; 
   }

   /* Preprocessing */
   mMinus1 = params.pattern_size - 1;
   m2 = params.pattern_size - 2;
   for(i=0; i<SIGMA; i++) B[i]=0;
   for (i = 1; i <= params.pattern_size; ++i)
      B[params.pattern[params.pattern_size-i]] |= (1<<(i-1));

   D = B[params.pattern[params.pattern_size-2]]; j=1; shift=0;
   if(D & (1<<(params.pattern_size-1))) shift = params.pattern_size-j;
   for(i=params.pattern_size-3; i>=0; i--) {
      D = (D<<1) & B[params.pattern[i]];
      j++;
      if(D & (1<<(params.pattern_size-1))) shift = params.pattern_size-j;
   }

   /* Searching */
   if( !memcmp(params.pattern,params.text,params.pattern_size) ) params.match[0] = 1;
   j = params.pattern_size;
   while (j < params.text_size) {
      D = (B[params.text[j]]<<1) & B[params.text[j-1]];
      if (D != 0) {
         pos = j;
         while (D=(D<<1) & B[params.text[j-2]])
            --j;
         j += m2;
         if (j == pos) {
            params.match[j] = 1;
            j+=shift;
         }
      }
      else j+=mMinus1;
   }
}

/*
 * Simplified Backward Nondeterministic DAWG Matching with loop unrolling designed for large patterns
 * The present implementation searches for prefixes of the pattern of length 32.
 * When an occurrence is found the algorithm tests for the whole occurrence of the pattern
 */

static void sbndm2_large(search_parameters params)
{
   unsigned int B[SIGMA], D;
   int i, j, pos, mMinus1, m2, p_len, shift;

   /* Preprocessing */
   p_len = params.pattern_size;
   params.pattern_size = 32;
   int diff = p_len-params.pattern_size;
   mMinus1 = params.pattern_size - 1;
   m2 = params.pattern_size - 2;
   for(i=0; i<SIGMA; i++) B[i]=0;
   for (i = 1; i <= params.pattern_size; ++i)
      B[params.pattern[params.pattern_size-i]] |= (1<<(i-1));
   D = B[params.pattern[params.pattern_size-1]]; j=1; shift=1;
   for(i=params.pattern_size-2; i>0; i--, j++) {
      if(D & (1<<(params.pattern_size-1))) shift = j;
      D = (D<<1) & B[params.pattern[i]];
   }

   /* Searching */
   if( !memcmp(params.pattern,params.text,params.pattern_size) ) params.match[0] = 1;
   j = params.pattern_size;
   while (j+diff < params.text_size) {
      D = (B[params.text[j]]<<1) & B[params.text[j-1]];
      if (D != 0) {
         pos = j;
         while (D=(D<<1) & B[params.text[j-2]])
            --j;
         j += m2;
         if (j == pos) {
            for(i=params.pattern_size+1; i<p_len && params.pattern[i]==params.text[j-params.pattern_size+1+i]; i++);
            if (i==p_len) params.match[j-params.pattern_size+1] = 1;
            j+=shift;
         }
      }
      else j+=mMinus1;
   }
}
