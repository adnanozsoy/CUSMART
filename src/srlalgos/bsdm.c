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
 * This is an implementation of the Backward SNR DAWG Matching algorithm
 * in S. Faro and T. Lecroq. 
 * A Fast Suffix Automata Based Algorithm for Exact Online String Matching. 
 * Implementation and Application of Automata - 17th International Conference, CIAA 2012, 
 * Lecture Notes in Computer Science, n.7381, pp.149--158, Springer-Verlag, Berlin, (2012).
 */

#include "include/define.h"
#include "bsdm.h"
#include <string.h>

void bsdm(search_parameters params) {
  unsigned int B[SIGMA];
  int i, j, k;
  unsigned int s,d;
  
  /* Preprocessing */
  unsigned int occ[SIGMA] = {0};
  int start = 0, len = 0;
  for (i=0, j=0; i<params.pattern_size; i++) {
    if (occ[params.pattern[i]]) {
      while(params.pattern[j]!=params.pattern[i]) {
	occ[params.pattern[j]]=0;
	j++;
      } 
      occ[params.pattern[j]]=0;
      j++;
    }
    occ[params.pattern[i]]=1;
    if (len < i-j+1 ) {
      len = i-j+1;
      start = j;
    }
  }
  unsigned int pos[SIGMA];
  for (i=0; i<SIGMA; i++) pos[i]=-1;
  for (i=0; i<len; i++) pos[params.pattern[start+i]]=i;
  
  //printf("%d / %d\n",m,len);
  //for (i=start; i<start+len; i++) printf("%d ",x[i]);
  //if (start+len<m) printf("[%d] ",x[start+len]);
  //printf("\n\n");
  
  /* Searching */
  for (i=0; i<params.pattern_size; i++) params.text[params.text_size+i]=params.pattern[i];
  unsigned char *xstart = params.pattern+start;
  int offset = len+start-1;
  j = len-1;
  while(j<params.text_size) {
    while ((i=pos[params.text[j]])<0) j+=len;
    k=1;
    while(k<=i && xstart[i-k]==params.text[j-k]) k++;
    if (k>i) {
      if (k==len) {
    	if (!memcmp(params.pattern,params.text+j-offset,params.pattern_size)) if (j-offset<=params.text_size-params.pattern_size) params.match[j] = 1;
      }
      else j-=k;
    }
    j+=len;
  }
}
