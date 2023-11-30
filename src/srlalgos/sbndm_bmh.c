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
 * This is an implementation of the Backward Nondeterministic DAWG Matching with Horspool Shift algorithm
 * in J. Holub and B. Durian.
 * Talk: Fast variants of bit parallel approach to suffix automata. The Second Haifa Annual International Stringology Research Workshop of the Israeli Science Foundation, (2005).
 */

#include "sbndm_bmh.h"
#include "include/define.h"

#include <stdlib.h>
#include <string.h>

static void sbndm_bmh_large(search_parameters params);

void sbndm_bmh(search_parameters params) {
   int j,i, last, first, hbc[SIGMA];
   unsigned int D, B[SIGMA], s;
   int mM1 = params.pattern_size-1;
   int mM2 = params.pattern_size-2;
   int count = 0, restore[XSIZE+1], shift;
   if(params.pattern_size>32) {
      sbndm_bmh_large(params);
      return;
   }

   /* preprocessing */
   for (i=0; i<SIGMA; i++)  B[i] = 0;
   for (i=0; i<params.pattern_size; i++) B[params.pattern[params.pattern_size-i-1]] |= (unsigned int)1 << (i+WORD-params.pattern_size);
   for (i=0;i<SIGMA;i++)   hbc[i]=2*params.pattern_size;
   for (i=0;i<params.pattern_size;i++) hbc[params.pattern[i]]=(2*params.pattern_size)-i-1;
   last = params.pattern_size;
   s = (unsigned int)(~0) << (WORD-params.pattern_size);
   s = (unsigned int)(~0);
   for (i=params.pattern_size-1; i>=0; i--) {
      s &= B[params.pattern[i]]; 
      if (s & ((unsigned int)1<<(WORD-1))) {
         if(i > 0)  last = i; 
      }
      restore[i] = last;
      s <<= 1;
   }
   shift = restore[0];

   for(i=0; i<params.pattern_size; i++) params.text[params.text_size+i]=params.pattern[i];

   /* Searching */
   if( !memcmp(params.pattern,params.text,params.pattern_size) ) params.match[0] = 1;
   i = params.pattern_size;
   while (1) {
      while ((D = B[params.text[i]]) == 0) i += hbc[params.text[i+params.pattern_size]];
      j = i-1; first = i-params.pattern_size+1;
      while (1) {
         D = (D << 1) & B[params.text[j]];
         if (!((j-first) && D)) break;
         j--;
      }
      if (D != 0) {
         if (i >= params.text_size) return;
         params.match[first] = 1;
         i += shift;
      } 
      else {
         i = j+params.pattern_size;
      }
   }
}

/*
 * Simplified Backward Nondeterministic DAWG Matching with Horspool shift designed for large patterns
 * The present implementation searches for prefixes of the pattern of length 32.
 * When an occurrence is found the algorithm tests for the whole occurrence of the pattern
 */

static void sbndm_bmh_large(search_parameters params) {
   int j,i, last, first, p_len, k, hbc[SIGMA];
   unsigned int D, B[SIGMA], s;
   int mM1 = params.pattern_size-1;
   int mM2 = params.pattern_size-2;
   int count = 0, restore[XSIZE+1], shift;

   p_len = params.pattern_size;
   params.pattern_size = 32;
   int diff = p_len-params.pattern_size;

   /* preprocessing */
   for (i=0; i<SIGMA; i++)  B[i] = 0;
   for (i=0; i<params.pattern_size; i++) B[params.pattern[params.pattern_size-i-1]] |= (unsigned int)1 << (i+WORD-params.pattern_size);
   for (i=0;i<SIGMA;i++)   hbc[i]=2*params.pattern_size;
   for (i=0;i<params.pattern_size;i++)     hbc[params.pattern[i]]=(2*params.pattern_size)-i-1;
   last = params.pattern_size;
   s = (unsigned int)(~0) << (WORD-params.pattern_size);
   s = (unsigned int)(~0);
   for (i=params.pattern_size-1; i>=0; i--) {
      s &= B[params.pattern[i]]; 
      if (s & ((unsigned int)1<<(WORD-1))) {
         if(i > 0)  last = i; 
      }
      restore[i] = last;
      s <<= 1;
   }
        shift = restore[0];

   for(i=0; i<params.pattern_size; i++) params.text[params.text_size+i]=params.pattern[i];

   /* Searching */
   if( !memcmp(params.pattern,params.text,p_len) ) params.match[0] = 1;
   i = params.pattern_size;
   while (1) {
      while ((D = B[params.text[i]]) == 0) i += hbc[params.text[i+params.pattern_size]];
      j = i-1; first = i-params.pattern_size+1;
      while (1) {
         D = (D << 1) & B[params.text[j]];
         if (!((j-first) && D)) break;
         j--;
      }
      if (D != 0) {
         if (i+diff >= params.text_size) return;
         k=params.pattern_size;
         while(k<p_len && params.pattern[k]==params.text[first+k]) k++;
         if(k==p_len) params.match[first] = 1;
         i += shift;
      } 
      else {
         i = j+params.pattern_size;
      }
   }
}
