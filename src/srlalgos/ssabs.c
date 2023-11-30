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
 * This is an implementation of the SSABS algorithm
 * in S. S. Sheik and S. K. Aggarwal and A. Poddar and N. Balakrishnan and K. Sekar.
 * A fast pattern matching algorithm. J. Chem. Inf. Comput., vol.44, pp.1251--1256, (2004).
 */

#include "include/define.h"
#include "ssabs.h"

void preQsBcSSABS(unsigned char *pattern, int pattern_size, int qbc[]) { 
  int i; 
  for (i=0;i<SIGMA;i++)	qbc[i]=pattern_size+1; 
  for (i=0;i<pattern_size;i++) qbc[pattern[i]]=pattern_size-i; 
} 

////////////Searching Phase/////////////////////////////////////// 
void ssabs(search_parameters params){ 
  int count,i,j =0; 
  int qsBc[SIGMA]; 
  unsigned char firstCh, lastCh; 
  
  preQsBcSSABS(params.pattern, params.pattern_size, qsBc); 
  firstCh = params.pattern[0]; 
  lastCh = params.pattern[params.pattern_size -1]; 
  for (i=0; i<params.pattern_size; i++) params.text[params.text_size+i]=lastCh; 
  while(j <= params.text_size - params.pattern_size){ 
    // Stage 1 
    if (lastCh == params.text[j + params.pattern_size - 1] && firstCh == params.text[j]) 
      { 
        //Stage 2 
        for (i = params.pattern_size-2; i > 0 && params.pattern[i] == params.text[j + i]; i--); 
        if (i <= 0) params.match[j] = 1; 
      } 
    // Stage 3 
    j += qsBc[params.text[j + params.pattern_size]]; 
  } 
} 
