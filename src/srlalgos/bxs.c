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
 * This is an implementation of the BNDM with Extended Shift algorithm
 * in B. Durian and H. Peltola and L. Salmela and J. Tarhio.
 * Bit-Parallel Search Algorithms for Long Patterns. 
 * Experimental Algorithms - 9th International Symposium, SEA 2010, Lecture Notes in Computer Science, n.6049, pp.129--140, Springer-Verlag, Berlin, (2010).
 */

#include "include/define.h"
#include "bxs.h"
#define Q 1

void bxs(search_parameters params) { 
  unsigned int B[SIGMA], D, set; 
  int i, j, first, k, mm, sh, m1; 
  if (params.pattern_size<Q) return; 
  //int larger = m>WORD? 1:0; 
  //if (larger) m = WORD; 
  int w = WORD, mq1 = params.pattern_size-Q+1, nq1 = params.text_size-Q+1; 
  if (w > params.pattern_size) w = params.pattern_size; 
  unsigned int mask = 1<<(w-1); 
  
  /* Preprocessing */
  set = 1; 
  for (i=0; i<SIGMA; i++) B[i]=0; 
  for (i = params.pattern_size-1; i >=0; i--) { 
    B[params.pattern[i]] |= set; 
    set<<=1; 
    if (set==0) set=1; 
  } 
  
  /* Searching */ 
  for (i=mq1-1; i<nq1; i+=mq1) { 
    D = B[params.text[i]]; 
    if ( D ) { 
      j = i;  
      first = i-mq1; 
      do { 
	j--; 
	if (D >= mask) { 
	  if (j-first) i=j; 
	  else { 
	    for (k=params.pattern_size; params.text[first+k]==params.pattern[k-1] && (k); k--); 
	    if ( k==0 ) params.match[i] = 1; 
	  } 
	  D = ((D<<1)|1) & B[params.text[j]]; 
	} 
	else D = (D<<1) & B[params.text[j]]; 
      } while (D && j>first); 
    } 
  } 
}
