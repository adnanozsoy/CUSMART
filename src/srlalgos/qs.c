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
 * This is an implementation of the Quick Search algorithm
 * in D. M. Sunday.
 * A very fast substring search algorithm. Commun. ACM, vol.33, n.8, pp.132--142, (1990).
 */

#include "qs.h"
#include "include/define.h"

void preQsBc(unsigned char *pattern, int pattern_size, int qbc[]) {
   int i;
   for(i=0;i<SIGMA;i++)   qbc[i]=pattern_size+1;
   for(i=0;i<pattern_size;i++) qbc[pattern[i]]=pattern_size-i;
}

void qs(search_parameters params) {
   int i, s, count, qsbc[SIGMA];
   
   /* Preoprocessing */
   preQsBc(params.pattern, params.pattern_size, qsbc);

   /* Searching */
   s = 0;
   count = 0;
   while(s <= params.text_size - params.pattern_size) {
      i = 0;
      while(i < params.pattern_size && params.pattern[i] == params.text[s+i]) i++;
      if(i == params.pattern_size) params.match[s] = 1;
      s += qsbc[params.text[s + params.pattern_size]];
   }
}
