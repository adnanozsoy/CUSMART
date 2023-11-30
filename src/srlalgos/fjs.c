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
 * This is an implementation of the Franek Jennings Smyth algorithm
 * in F. Franek and C. G. Jennings and W. F. Smyth.
 * A simple fast hybrid pattern-matching algorithm. J. Discret. Algorithms, vol.5, pp.682--695, (2007).
 */

#include "include/define.h"
#include "fjs.h"
#include <stdlib.h>

void preQsBcFJS(unsigned char *pattern, int pattern_size, int qbc[]) {
   int i;
   for (i=0;i<SIGMA;i++)   qbc[i]=pattern_size+1;
   for (i=0;i<pattern_size;i++) qbc[pattern[i]]=pattern_size-i;
}

void preKmpFJS(unsigned char *pattern, int pattern_size, int kmpNexy[]) {
   int i, j;
   i = 0;
   j = kmpNexy[0] = -1;
   while (i < pattern_size) {
      while (j > -1 && pattern[i] != pattern[j])
         j = kmpNexy[j];
      i++;
      j++;
      if (i<pattern_size && pattern[i] == pattern[j])
         kmpNexy[i] = kmpNexy[j];
      else
         kmpNexy[i] = j;
   }
}

void fjs(search_parameters params) {
   int i, s, qsbc[SIGMA], kmp[XSIZE];

   /* Preprocessing */
   preQsBcFJS(params.pattern,params.pattern_size,qsbc);
   preKmpFJS(params.pattern,params.pattern_size,kmp);

   /* Searching */
   s = 0;
 
   while(s<=params.text_size-params.pattern_size) {
      while(s<=params.text_size-params.pattern_size &&
	    params.pattern[params.pattern_size-1]!=params.text[s+params.pattern_size-1]) s+=qsbc[params.text[s+params.pattern_size]];
      if (s>params.text_size-params.pattern_size) {
	return;
      }
      i=0; 
      while(i<params.pattern_size && params.pattern[i]==params.text[s+i]) i++;
      if (i>=params.pattern_size) params.match[s] = 1;
      s+=(i-kmp[i]);
   }
}

