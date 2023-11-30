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
 * This is an implementation of the Backward Nondeterministic DAWG Matching with loop unrolling algorithm
 * in J. Holub and B. Durian.
 * Talk: Fast variants of bit parallel approach to suffix automata. 
 * The Second Haifa Annual International Stringology Research Workshop of the Israeli Science Foundation, (2005).
 */

#include "bndmq2.h"
#include "include/define.h"

#include <stdlib.h>
#include <string.h>

void bndmq2(search_parameters params) {
   unsigned int D, B[SIGMA], M, s;
   int i,j,q, first;
   q = 2;
   if(params.pattern_size<q) return;
   if(params.pattern_size>WORD) {
      bndmq2_large(params);
      return;
   }

   /* Preprocessing */
   for (i=0; i<SIGMA; i++) B[i] = 0;
   s=1;
   for (i=params.pattern_size-1; i>=0; i--){
      B[params.pattern[i]] |= s;
      s <<= 1;
   }
   for (j=0; j<params.pattern_size; j++) params.text[params.text_size+j]=params.pattern[j];
   M = 1 << (params.pattern_size-1);

   /* Searching */
   if(!memcmp(params.pattern,params.text,params.pattern_size)) params.match[0] = 1;
   i = params.pattern_size+1-q;
   while (i <= params.text_size - q) {
      D = (B[params.text[i+1]]<<1)&B[params.text[i]]; // GRAM2
      if (D != 0) {
         j = i;
         first = i - (params.pattern_size - q);
         do {
            if ( D >= M ) {
               if (j > first) i = j-1;
               else params.match[first] = 1;
            }
            j = j-1;
            D = (D<<1) & B[params.text[j]];
         } while (D != 0);
      }
      i = i+params.pattern_size-q+1;
   }
}

/*
 * Backward Nondeterministic DAWG Matching using q-grams designed for large patterns
 * The present implementation searches for prefixes of the pattern of length 32.
 * When an occurrence is found the algorithm tests for the whole occurrence of the pattern
 */

void bndmq2_large(search_parameters params) {
   unsigned int D, B[SIGMA], M, s;
   int i, j, q, first, p_len, k;
   q = 2;
   if(params.pattern_size<=q) return;

   p_len = params.pattern_size;
   params.pattern_size = 32;

   /* Preprocessing */
   for (i=0; i<SIGMA; i++) B[i] = 0;
   s=1;
   for (i=params.pattern_size-1; i>=0; i--){
       B[params.pattern[i]] |= s;
       s <<= 1;
   }
   for (j=0; j<params.pattern_size; j++) params.text[params.text_size+j]=params.pattern[j];
   M = 1 << (params.pattern_size-1);

   /* Searching */
   if(!memcmp(params.pattern,params.text,p_len)) params.match[0] = 1;
   i = params.pattern_size+1-q;
   while (i <= params.text_size - q) {
      D = (B[params.text[i+1]]<<1)&B[params.text[i]]; // GRAM2
      if (D != 0) {
         j = i;
         first = i - (params.pattern_size - q);
         do {
            if ( D >= M ) {
               if (j > first) i = j-1;
               else {
                  k = params.pattern_size;
                  while(k<p_len && params.pattern[k]==params.text[first+k]) k++;
                  if(k==p_len) params.match[first] = 1;
               }
            }
            j = j-1;
            D = (D<<1) & B[params.text[j]];
         } while (D != 0);
      }
      i = i+params.pattern_size-q+1;
   }
}


