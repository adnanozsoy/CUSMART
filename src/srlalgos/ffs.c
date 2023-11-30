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
 * This is an implementation of the Forward Fast Search algorithm
 * in D. Cantone and S. Faro.
 * Fast-Search Algorithms: New Efficient Variants of the Boyer-Moore Pattern-Matching Algorithm. J. Autom. Lang. Comb., vol.10, n.5/6, pp.589--608, (2005).
 */

#include "ffs.h"
#include "include/define.h"
#include <stdlib.h>
#include <string.h>

static void Forward_Suffix_Function(unsigned char *x, int m, int bm_gs[XSIZE][SIGMA], int alpha) {
   int init;
   int i, j, last, suffix_len, temx[XSIZE];
   init = 0;
   for(i=0;i<m;i++) for(j=init; j<init+alpha;j++) bm_gs[i][j] = m+1;
   for(i=0; i<m; i++) temx[i]=i-1;
   for(suffix_len=0; suffix_len<=m; suffix_len++) {
      last = m-1;
      i = temx[last];
      while(i>=0) {
         if( bm_gs[m-suffix_len][x[i+1]]>m-1-i )
            if(i-suffix_len<0 || (i-suffix_len>=0 && x[i-suffix_len]!=x[m-1-suffix_len]))
               bm_gs[m-suffix_len][x[i+1]]=m-1-i;
         if((i-suffix_len >= 0 && x[i-suffix_len]==x[last-suffix_len]) || (i-suffix_len <  0)) {
            temx[last]=i;
            last=i;
         }
         i = temx[i];
      }
      if(bm_gs[m-suffix_len][x[0]] > m) bm_gs[m-suffix_len][x[0]] = m;
      temx[last]=-1;
   }
}

void ffs0(search_parameters params) {
   int i, j, k, s, count, gs[XSIZE][SIGMA], bc[SIGMA];
   char ch = params.pattern[params.pattern_size-1];

   /* Preprocessing */
   Forward_Suffix_Function(params.pattern, params.pattern_size, gs, SIGMA);
   for (i=0; i < SIGMA; i++) bc[i]=params.pattern_size;
   for (j=0; j < params.pattern_size; j++) bc[params.pattern[j]]=params.pattern_size-j-1;
   for(i=0; i<=params.pattern_size; i++) params.text[params.text_size+i]=ch;
   params.text[params.text_size+params.pattern_size]='\0';

   /* Searching */
   if( !memcmp(params.pattern,params.text,params.pattern_size) ) count++; 
   s=params.pattern_size;
   while(s<params.text_size) {
      while(k=bc[params.text[s]])   s += k;
      for(j=s-1, k=params.pattern_size-1; k>0 && params.pattern[k-1]==params.text[j]; k--, j--);
      if(!k && s<params.text_size) params.match[j] = 1;
      s += gs[k][params.text[s+1]];
   }
}



