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
 * This is an implementation of the The Backward Fast Search, Fast Boyer Moore algorithm
 * in D. Cantone and S. Faro.
 * Fast-Search Algorithms: New Efficient Variants of the Boyer-Moore Pattern-Matching Algorithm. 
 * J. Autom. Lang. Comb., vol.10, n.5/6, pp.589--608, (2005).
 */

#include "bfs.h"
#include "include/define.h"
#include <stdlib.h>
#include <string.h>

static void PreBFS(unsigned char *pattern, int pattern_size, int *bm_gs) {
   int i, j, c, last, suffix_len, temp[XSIZE];
   suffix_len = 1;
   last = pattern_size-1;
   for(i=0;i<=pattern_size;i++) for(j=0; j<SIGMA;j++) bm_gs[SIGMA * i + j] = pattern_size;
   for(i=0; i<=pattern_size; i++) temp[i]=-1;
   for(i=pattern_size-2; i>=0; i--)
   if(pattern[i]==pattern[last]) {
      temp[last]=i;
      last = i;
   }
   suffix_len++;
   while(suffix_len<=pattern_size) {
      last = pattern_size-1;
      i = temp[last];
      while(i>=0) {
         if(i-suffix_len+1>=0) {
            if(pattern[i-suffix_len+1]==pattern[last-suffix_len+1]) {
               temp[last]=i;
               last=i;
            }
            if(bm_gs[SIGMA*(pattern_size-suffix_len+1) + pattern[i-suffix_len+1]] > pattern_size-1-i)
               bm_gs[SIGMA*(pattern_size-suffix_len+1) + pattern[i-suffix_len+1]] = pattern_size-1-i;
         }
         else {
            temp[last]=i;
            last = i;
            for(c=0; c<SIGMA; c++)
               if(bm_gs[SIGMA*(pattern_size-suffix_len+1) + c] > pattern_size-1-i)
                  bm_gs[SIGMA*(pattern_size-suffix_len+1) + c] = pattern_size-1-i;
         }
         i = temp[i];
      }
      temp[last]=-1;
      suffix_len++;
   }
}

void bfs(search_parameters params)
{
   int i, j, k, s, bc[SIGMA];
   int *last, first;
   char ch = params.pattern[params.pattern_size-1];
   int* gs = (int*)malloc(SIGMA * (params.pattern_size+1) * sizeof(int));

   /* Preprocessing */    
   for(i=0;i<SIGMA;i++) bc[i]=params.pattern_size;
   for(i=0;i<params.pattern_size;i++) bc[params.pattern[i]]=params.pattern_size-i-1;
   for(i=0;i<params.pattern_size;i++) params.text[params.text_size+i]=params.pattern[i];
   PreBFS(params.pattern, params.pattern_size, gs);

   /* Searching */   
   s=params.pattern_size-1;
   last = gs[SIGMA * params.pattern_size];
   first = gs[SIGMA + params.pattern[0]];
   while(s<params.text_size) {
      while((k=bc[params.text[s]])!=0)   s = s + k;
      for(j=s-1, k=params.pattern_size-1; k>0 && params.pattern[k-1]==params.text[j]; k--, j--);
      if( k==0 ) {
         if(s<params.text_size) params.match[j] = 1;
         s += first;
      }
      else s += gs[SIGMA * k + params.text[j]];
   }
}


