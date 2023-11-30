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
 * This is an implementation of the Boyer Moore Horspoolusing Probabilities algorithm
 * in M. E. Nebel.
 * Fast string matching by using probabilities: on an optimal mismatch variant of Horspool's algorithm.
 * Theor. Comput. Sci., vol.359, n.1, pp.329--343, Elsevier Science Publishers Ltd., Essex, UK, (2006).
 */

#include "pbmh.h"
#include "include/define.h"

#include "stdlib.h"

void pbmh(search_parameters params) {
  int i, j, s, tmp, count, hbc[SIGMA], v[XSIZE], FREQ[SIGMA];
  
  /* Computing the frequency of characters */
  for(i=0; i<SIGMA; i++)	FREQ[i] = 0;
  for(i=0; i<100; i++) FREQ[params.text[i]]++;
  
  /* Preprocessing */
  for (i=0; i<params.pattern_size; i++) v[i]=i;
  for (i=params.pattern_size-1; i>0; i--)
    for (j=0; j<i; j++)
      if (FREQ[params.pattern[v[j]]]>FREQ[params.pattern[v[j+1]]]) {   
	tmp = v[j+1];
	v[j+1] = v[j];
	v[j] = tmp;
      }
  for (i=0;i<SIGMA;i++)   hbc[i]=params.pattern_size;
  for (i=0;i<params.pattern_size-1;i++) hbc[params.pattern[i]]=params.pattern_size-i-1;
  
  /* Searching */
  s = 0;
  while(s<=params.text_size-params.pattern_size) {
    i=0;
    while(i<params.pattern_size && params.pattern[v[i]]==params.text[s+v[i]]) i++;
    if (i==params.pattern_size) params.match[s] = 1;
    s+=hbc[params.text[s+params.pattern_size-1]];
  }
}
