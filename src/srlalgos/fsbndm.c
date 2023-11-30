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
 * contact the authors at: faro@dmi.unict.it, thierry.lecroq@univ-rouen.fr
 * download the tool at: http://www.dmi.unict.it/~faro/smart/
 *
 * This is an implementation of the Forward Semplified BNDM algorithm
 * in S. Faro and T. Lecroq. 
 * Efficient Variants of the Backward-Oracle-Matching Algorithm. 
 * Proceedings of the Prague Stringology Conference 2008, pp.146--160, Czech Technical University in Prague, Czech Republic, (2008).
 */

#include "fsbndm.h"
#include "include/define.h"

#include <stdlib.h>
#include <string.h>

/*
 * Forward Semplified BNDM algorithm designed for large patterns
 * The present implementation searches for prefixes of the pattern of length 32.
 * When an occurrence is found the algorithm tests for the whole occurrence of the pattern
 */

static void fsbndm_large(search_parameters params) 
{
   unsigned int B[SIGMA], D, set;
   int i, j, pos, mMinus1, p_len, k,s;
   
   p_len = params.pattern_size;
   params.pattern_size = 30;

   /* Preprocessing */
   mMinus1 = params.pattern_size - 1;
   set = 1;
   for(i=0; i<SIGMA; i++) B[i]=set;
   for (i = 0; i < params.pattern_size; ++i) B[params.pattern[i]] |= (1<<(params.pattern_size-i));

   /* Searching */
   if(!memcmp(params.pattern,params.text,params.pattern_size)) params.match[0] = 1;
   j = params.pattern_size;
   while (j < params.text_size) {
      D = (B[params.text[j+1]]<<1) & B[params.text[j]];
      if (D != 0) {
         pos = j;
         while (D=(D<<1) & B[params.text[j-1]]) --j;
         j += mMinus1;
         if (j == pos) {
            k = params.pattern_size; s=j-mMinus1;
            while (k<p_len && params.pattern[k]==params.text[s+k]) k++;
            if (k==p_len) params.match[s] = 1;
            ++j;
         }
      }
      else j+=params.pattern_size;
   }
}

void fsbndm(search_parameters params) {
   unsigned int B[SIGMA], D, set;
   int i, j, pos, mMinus1;
   
   if(params.pattern_size>31) {
      fsbndm_large(params);
      return;
   }

   /* Preprocessing */
   mMinus1 = params.pattern_size - 1;
   set = 1;
   for(i=0; i<SIGMA; i++) B[i]=set;
   for (i = 0; i < params.pattern_size; ++i) B[params.pattern[i]] |= (1<<(params.pattern_size-i));

   /* Searching */
   if(!memcmp(params.pattern,params.text,params.pattern_size)) params.match[0] = 1;
   j = params.pattern_size;
   while (j < params.text_size) {
      D = (B[params.text[j+1]]<<1) & B[params.text[j]];
      if (D != 0) {
         pos = j;
         while (D=(D<<1) & B[params.text[j-1]]) --j;
         j += mMinus1;
         if (j == pos) {
            params.match[j] = 1;
            ++j;
         }
      }
      else j+=params.pattern_size;
   }
}
