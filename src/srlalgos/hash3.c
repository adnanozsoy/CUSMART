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
 * This is an implementation of the Wu Manber for Single Pattern Matching algorithm
 * in T. Lecroq.
 * Fast exact string matching algorithms. ipl, vol.102, n.6, pp.229--235, Elsevier North-Holland, Inc., Amsterdam, The Netherlands, The Netherlands, (2007).
 */

#include "hash3.h"
#include "include/define.h"
#include <string.h>
#include <stdlib.h>

#define RANK3 3

void hash3(search_parameters params) { 
   int j, i, sh, sh1, mMinus1, mMinus2, shift[WSIZE]; 
   unsigned char h; 
   if (params.pattern_size<3) return; 
   mMinus1 = params.pattern_size-1; 
   mMinus2 = params.pattern_size-2; 

   /* Preprocessing */ 
   for (i = 0; i < WSIZE; ++i) 
      shift[i] = mMinus2; 

   h = params.pattern[0]; 
   h = ((h<<1) + params.pattern[1]); 
   h = ((h<<1) + params.pattern[2]); 
   shift[h] = params.pattern_size-RANK3; 
   for (i=RANK3; i < mMinus1; ++i) { 
      h = params.pattern[i-2]; 
      h = ((h<<1) + params.pattern[i-1]); 
      h = ((h<<1) + params.pattern[i]); 
      shift[h] = mMinus1-i; 
   } 
   h = params.pattern[i-2]; 
   h = ((h<<1) + params.pattern[i-1]); 
   h = ((h<<1) + params.pattern[i]); 
   sh1 = shift[h]; 
   shift[h] = 0; 
   if (sh1==0) sh1=1; 


   /* Searching */ 
   i = mMinus1; 
   memcpy(params.text+params.text_size, params.pattern, params.pattern_size); 
   while (1) { 
      sh = 1; 
      while (sh != 0) { 
         h = params.text[i-2]; 
         h = ((h<<1) + params.text[i-1]); 
         h = ((h<<1) + params.text[i]); 
         sh = shift[h]; 
         i+=sh; 
      } 
      if (i < params.text_size) { 
         j=0; 
         while(j<params.pattern_size && params.pattern[j]==params.text[i-mMinus1+j]) j++; 
         if (j>=params.pattern_size) { 
            params.match[i-mMinus1] = 1; 
         } 
         i+=sh1; 
      } 
      else return; 
   } 
} 
