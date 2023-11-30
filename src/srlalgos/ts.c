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
 * This is an implementation of the Tailed Substring algorithm
 * in D. Cantone and S. Faro.
 * Searching for a substring with constant extra-space complexity. Proc. of Third International Conference on Fun with algorithms, pp.118--131, (2004).
 */

#include "include/define.h"
#include "ts.h"

void ts(search_parameters params) { 
   int s, j, i, k, h, dim; 

   /* Searching */ 

   /* phase n.1*/ 
   s = 0; i = params.pattern_size-1; k = params.pattern_size-1; dim = 1; 
   while (s <= params.text_size-params.pattern_size && i-dim >= 0) { 
   	  if (params.pattern[i] != params.text[s+i]) s++; 
      else { 
      	 for (j=0; j<params.pattern_size && params.pattern[j]==params.text[s+j]; j++); 
         if (j==params.pattern_size) params.match[s] = 1; 
         for (h=i-1; h>=0 && params.pattern[h]!=params.pattern[i]; h--); 
         if (dim<i-h) {k=i; dim=i-h;} 
         s+=i-h; 
         i--; 
      } 
   } 

   /* phase n.2 */ 
   while (s <= params.text_size - params.pattern_size) { 
   	  if (params.pattern[k]!=params.text[s+k]) s++; 
      else { 
      	 j=0; 
         while(j<params.pattern_size && params.pattern[j]==params.text[s+j]) j++; 
         if (j==params.pattern_size) params.match[s] = 1; 
         s+=dim; 
      } 
   } 
} 
