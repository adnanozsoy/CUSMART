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
 * This is an implementation of the Forward Backward Oracle Matching algorithm
 * in S. Faro and T. Lecroq.
 * Efficient Variants of the Backward-Oracle-Matching Algorithm. 
 * Proceedings of the Prague Stringology Conference 2008, pp.146--160, Czech Technical University in Prague, Czech Republic, (2008).
 */

#include "fbom.h"
#include "include/define.h"

#include <stdlib.h>
#include <string.h>

void fbom(search_parameters params) { 
   int S[XSIZE], FT[SIGMA][SIGMA]; 
   int *trans[XSIZE]; 
   int i, j, p, q; 
   int iMinus1, mMinus1, count; 
   unsigned char c; 

   /* Preprocessing */ 
   for (i=0; i<=params.pattern_size+1; i++) trans[i] = (int *)malloc (sizeof(int)*(SIGMA)); 
   for (i=0; i<=params.pattern_size+1; i++) for (j=0; j<SIGMA; j++) trans[i][j]=UNDEFINED; 
   S[params.pattern_size] = params.pattern_size + 1; 
   for (i = params.pattern_size; i > 0; --i) { 
      iMinus1 = i - 1; 
      c = params.pattern[iMinus1]; 
      trans[i][c] = iMinus1; 
      p = S[i]; 
      while (p <= params.pattern_size && (q = trans[p][c]) ==  UNDEFINED) { 
         trans[p][c] = iMinus1; 
         p = S[p]; 
      } 
      S[iMinus1] = (p == params.pattern_size + 1 ? params.pattern_size : q); 
   } 

   /* Construct the First transition table */ 
   for (i=0; i<SIGMA; i++) { 
      q = trans[params.pattern_size][i]; 
      for (j=0; j<SIGMA; j++) 
         if (q>=0) { 
            if ((p=trans[q][j])>=0) FT[i][j] = p; 
            else FT[i][j]=params.pattern_size+params.pattern_size+1; 
         } 
         else FT[i][j] = params.pattern_size+params.pattern_size+1; 
   } 
   q = trans[params.pattern_size][params.pattern[params.pattern_size-1]]; 
   for (i=0; i<SIGMA; i++) FT[i][params.pattern[params.pattern_size-1]] = q; 
   for (i=0; i<SIGMA; i++) if (FT[params.pattern[0]][i]>params.pattern_size) FT[params.pattern[0]][i]-=1; 

   /* Searching */ 
   for (i=0; i<params.pattern_size; i++) params.text[params.text_size+i]=params.pattern[i]; 
   if ( !memcmp(params.pattern,params.text,params.pattern_size) ) params.match[i] = 1; 
   j=params.pattern_size; 
   mMinus1 = params.pattern_size-1; 
   while (j<params.text_size) { 
      while ( (p=FT[params.text[j+1]][params.text[j]])>params.pattern_size ) j+=p-params.pattern_size; 
        i = j-1; 
        while ( (p = trans[p][params.text[i]]) != UNDEFINED ) i--; 
        if (i < j-mMinus1 && j<params.text_size) { 
           params.match[j] = 1; 
           i++; 
        } 
        j = i + params.pattern_size; 
   } 

   for (i=0; i<=params.pattern_size+1; i++) free(trans[i]); 
}
