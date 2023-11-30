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
 * This is an implementation of the Fast Search algorithm
 * in D. Cantone and S. Faro.
 * Fast-Search: a new efficient variant of the Boyer-Moore string matching algorithm. 
 * WEA 2003, Lecture Notes in Computer Science, vol.2647, pp.247--267, Springer-Verlag, Berlin, (2003).
 */

#include "fs.h"
#include "include/define.h"

#include "stdio.h"
#include "stdlib.h"
#include "string.h"


void Pre_GS(unsigned char *x, int m, int bm_gs[]) {
   int i, j, p, f[XSIZE];
   for (i=0;i<XSIZE;i++) bm_gs[i]=0;
   f[m]=j=m+1;
   for (i=m; i > 0; i--) {
      while (j <= m && x[i-1] != x[j-1]) {
         if (bm_gs[j] == 0) bm_gs[j]=j-i;
         j=f[j];
      }
      f[i-1]=--j;
   }
   p=f[0];
   for (j=0; j <= m; ++j) {
      if (bm_gs[j] == 0) bm_gs[j]=p;
      if (j == p) p=f[p];
   }
}

void fs(search_parameters params) {
	int a,i, j, k, s, bc[SIGMA], gs[XSIZE];
	char ch = params.pattern[params.pattern_size-1];

	/* Preprocessing */
	for (a=0; a < SIGMA; a++) 
		bc[a]=params.pattern_size;
		
	for (j=0; j < params.pattern_size; j++) 
		bc[params.pattern[j]]=params.pattern_size-j-1;
		
	Pre_GS(params.pattern, params.pattern_size, gs);
	
	//for (i=0; i<params.pattern_size; i++) params.text[params.text_size+i]=ch;

	/* Searching */

	//if ( !memcmp(params.pattern, params.text, params.pattern_size) ) params.match[j] = 1; 
	//Because we start from pattern_size-1 the above memcmp is unnecessary.
	s = params.pattern_size - 1;
	while(s < params.text_size) {
		while((k=bc[params.text[s]]))   
			s += k;
		j=2;
		while (j <= params.pattern_size && params.pattern[params.pattern_size-j] == params.text[s-j+1]) 
			j++;
			
		if ( j > params.pattern_size && s < params.text_size) 
			params.match[s-j+2] = 1;
			
		s += gs[params.pattern_size-j+1];
	}
  
}
