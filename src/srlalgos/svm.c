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
 * This is an implementation of the Shift Vector Matching algorithm
 * in H. Peltola and J. Tarhio.
 * Alternative Algorithms for Bit-Parallel String Matching.
 * Proceedings of the 10th International Symposium on String Processing and Information Retrieval SPIRE'03, (2003).
 */

#include "include/define.h"
#include "svm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Shift Vector Matching algorithm designed for large patterns
 * The present implementation searches for prefixes of the pattern of length 32.
 * When an occurrence is found the algorithm tests for the whole occurrence of the pattern
 */

static void svm_large(search_parameters params) {
   int count, s, j, first, k, p_len;
   unsigned int tmp, h, sv, cv[SIGMA], ONE;
   p_len = params.pattern_size;
   params.pattern_size = 32;   

   /* Preprocessing */
   ONE = 1;
   tmp = (~0);
   tmp >>= (32-params.pattern_size);
   for(j = 0; j < SIGMA; j++) cv[j] = tmp;
   tmp = ~ONE;
   for(j = params.pattern_size-1; j >= 0; j--) {
      cv[params.pattern[j]] &= tmp;
      tmp <<= 1;
      tmp |= 1;
   }
   

   /* Searching */
   sv = 0;   
   count = 0;
   if( !memcmp(params.pattern,params.text,params.pattern_size) ) params.match[0] = 1;
   s = params.pattern_size;
   while(s < params.text_size){
      sv |= cv[params.text[s]];
      j = 1;
      while((sv & ONE) == 0) {
         sv |= (cv[params.text[s-j]] >> j);
         if(j >= params.pattern_size) {
            k = params.pattern_size; first = s-params.pattern_size+1;
            while (k<p_len && params.pattern[k]==params.text[first+k]) k++;
            if (k==p_len) params.match[first] = 1; 
            break;
         }
         j++;
      }
      sv >>= 1; s += 1;
      while((sv & ONE)==1) {   
         sv >>= 1;
         s += 1;
      }
   }
}


void svm(search_parameters params) {
   int count, s, j;
   unsigned int tmp, h, sv, cv[SIGMA], ONE;

   if(params.pattern_size>32) {
      svm_large(params);
      return;
   }

   /* Preprocessing */
   ONE = 1;
   tmp = (~0);
   tmp >>= (32-params.pattern_size);
   for(j = 0; j < SIGMA; j++) cv[j] = tmp;
   tmp = ~ONE;
   for(j = params.pattern_size-1; j >= 0; j--) {
      cv[params.pattern[j]] &= tmp;
      tmp <<= 1;
      tmp |= 1;
   }
   

   /* searching */
   sv = 0;   
   if( !memcmp(params.pattern,params.text,params.pattern_size) ) params.match[0] = 1;
   s = params.pattern_size;
   while(s < params.text_size){
      sv |= cv[params.text[s]];
      j = 1;
      while((sv & ONE) == 0) {
         sv |= (cv[params.text[s-j]] >> j);
         if(j >= params.pattern_size) {params.match[s] = 1; break;}
         j++;
      }
      sv >>= 1; s += 1;
      while((sv & ONE)==1) {   
         sv >>= 1;
         s += 1;
      }
   }
}

