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
 * This is an implementation of the Shift Or algorithm
 * in R. Baeza-Yates and G. H. Gonnet.
 * A new approach to text searching. Commun. ACM, vol.35, n.10, pp.74--82, ACM, New York, NY, USA, (1992).
 */

#include "include/define.h"
#include "so.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"

int preSo(unsigned char *pattern, int pattern_size, unsigned int S[]) { 
   unsigned int j, lim; 
   int i; 
   for (i = 0; i < SIGMA; ++i) 
      S[i] = ~0; 
   for (lim = i = 0, j = 1; i < pattern_size; ++i, j <<= 1) { 
      S[pattern[i]] &= ~j; 
      lim |= j; 
   } 
   lim = ~(lim>>1); 
   return(lim); 
} 

/*
 * Shift Or algorithm designed for large patterns
 * The present implementation searches for prefixes of the pattern of length 32.
 * When an occurrence is found the algorithm tests for the whole occurrence of the pattern
 */
void so_large(search_parameters params) { 
   unsigned int lim, D,k,h,p_len; 
   unsigned int S[SIGMA]; 
   int j, count; 

   p_len = params.pattern_size;
   params.pattern_size = 32;

   /* Preprocessing */ 
   lim = preSo(params.pattern, params.pattern_size, S); 

   /* Searching */ 
   count = 0;
   for (D = ~0, j = 0; j < params.text_size; ++j) { 
      D = (D<<1) | S[params.text[j]]; 
      if (D < lim) {
         k = 0;
         h = j - params.pattern_size + 1;
         while(k<p_len && params.pattern[k] == params.text[h+k]) k++;
         if (k==p_len) params.match[j - params.pattern_size + 1] = 1; 
      }
   } 
} 

void so(search_parameters params) { 
   unsigned int lim, D; 
   unsigned int S[SIGMA]; 
   int j, count; 
   if (params.pattern_size > WORD) {
     so_large(params);
     return;
   }
   /* Preprocessing */ 
   lim = preSo(params.pattern, params.pattern_size, S); 

   /* Searching */ 
   count = 0;
   for (D = ~0, j = 0; j < params.text_size; ++j) { 
      D = (D<<1) | S[params.text[j]]; 
      if (D < lim)
	params.match[j - params.pattern_size + 1] = 1;
   } 
} 

