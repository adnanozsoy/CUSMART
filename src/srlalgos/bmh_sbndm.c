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
 * This is an implementation of the Horspool with BNDM test algorithm
 * in J. Holub and B. Durian. 
 * Talk: Fast variants of bit parallel approach to suffix automata. 
 * The Second Haifa Annual International Stringology Research Workshop of the Israeli Science Foundation, (2005).
 */

#include "include/define.h"
#include "bmh_sbndm.h"
#include "stdlib.h"
#include "string.h"


static void bmh_sbndm_large(search_parameters params) {
   int i, j,k,s, hbc[SIGMA], shift, p_len;
   unsigned int B[SIGMA], D;

   p_len = params.pattern_size;
   params.pattern_size = 32;
   int diff = p_len-params.pattern_size;

   /* Preprocessing */
   for(i=0;i<SIGMA;i++)
      hbc[i]=params.pattern_size;
   for(i=0;i<params.pattern_size;i++)
      hbc[params.pattern[i]]=params.pattern_size-i-1;
   for (i=0; i<SIGMA; i++)  
      B[i] = 0;
   for (i=0; i<params.pattern_size; i++) 
      B[params.pattern[params.pattern_size-i-1]] |= (unsigned int)1 << (i+WORD-params.pattern_size);
   for(i=0; i<params.pattern_size; i++) 
      params.text[params.text_size+i]=params.pattern[i];
   D = B[params.pattern[params.pattern_size-1]]; j=1; shift=1;
   for(i=params.pattern_size-2; i>0; i--, j++) {
      if(D & (1<<(params.pattern_size-1))) shift = j;
      D = (D<<1) & B[params.pattern[i]];
   }

   /* Searching */      
   if( !memcmp(params.pattern,params.text,params.pattern_size) ) params.match[0] = 1;
   i = params.pattern_size;
   while(i+diff < params.text_size) {
      while( (k=hbc[params.text[i]])!=0 ) i+=k;
      j=i; s=i-params.pattern_size+1;
      D = B[params.text[j]];
      while(D!=0) { 
         j--;
         D = (D<<1) & B[params.text[j]];
      }
      if(j<s) {
         if(s<params.text_size) {
            k = params.pattern_size;
            while(k<p_len && params.pattern[k]==params.text[s+k]) k++;
            if(k==p_len && i+diff<params.text_size) params.match[s] = 1;
         }
         i += shift;
      }
      else i = j+params.pattern_size;
   }
}



void bmh_sbndm(search_parameters params) {
   int i, j,k,s, hbc[SIGMA], shift;
   unsigned int B[SIGMA], D;

   if (params.pattern_size>32) {
      bmh_sbndm_large(params);
      return;
   }

   /* Preprocessing */
   for(i=0;i<SIGMA;i++) 
      hbc[i]=params.pattern_size;
   for(i=0;i<params.pattern_size;i++) 
      hbc[params.pattern[i]]=params.pattern_size-i-1;
   for (i=0; i<SIGMA; i++)  
      B[i] = 0;
   for (i=0; i<params.pattern_size; i++) 
      B[params.pattern[params.pattern_size-i-1]] |= (unsigned int)1 << (i+WORD-params.pattern_size);
   for(i=0; i<params.pattern_size; i++) 
      params.text[params.text_size+i]=params.pattern[i];
   D = B[params.pattern[params.pattern_size-1]]; j=1; shift=1;
   for(i=params.pattern_size-2; i>0; i--, j++) {
      if(D & (1<<(params.pattern_size-1))) shift = j;
      D = (D<<1) & B[params.pattern[i]];
   }

   /* Searching */      
   if( !memcmp(params.pattern,params.text,params.pattern_size) ) params.match[0] = 1;
   i = params.pattern_size;
   while(i < params.text_size) {
      while( (k=hbc[params.text[i]])!=0 ) i+=k;
      j=i; s=i-params.pattern_size+1;
      D = B[params.text[j]];
      while(D!=0) { 
         j--;
         D = (D<<1) & B[params.text[j]];
      }
      if(j<s) {
         if(s<params.text_size && i<params.text_size) params.match[s] = 1;
         i += shift;
      }
      else i = j+params.pattern_size;
   }
}

/*
 * Horspool algorithm with BNDM test designed for large patterns
 * The present implementation searches for prefixes of the pattern of length 32.
 * When an occurrence is found the algorithm tests for the whole occurrence of the pattern
 */
