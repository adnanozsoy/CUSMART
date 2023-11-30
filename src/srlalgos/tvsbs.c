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
 * This is an implementation of the TVSBS algorithm
 * in R. Thathoo and A. Virmani and S. S. Lakshmi and N. Balakrishnan and K. Sekar.
 * TVSBS: A Fast Exact Pattern Matching Algorithm for Biological Sequences. J. Indian Acad. Sci., Current Sci., vol.91, n.1, pp.47--53, (2006).
 */

#include "tvsbs.h"
#include "include/define.h"

#include "stdio.h"
#include "stdlib.h"

void TVSBSpreBrBc(unsigned char *pattern, int pattern_size, int brBc[SIGMA][SIGMA]) {
   int a, b, i;
   for (a = 0; a < SIGMA; ++a)
      for (b = 0; b < SIGMA; ++b)
         brBc[a][b] = pattern_size + 2;
   for (a = 0; a < SIGMA; ++a)
      brBc[a][pattern[0]] = pattern_size + 1;
   for (i = 0; i < pattern_size - 1; ++i)
      brBc[pattern[i]][pattern[i + 1]] = pattern_size - i;
   for (a = 0; a < SIGMA; ++a)
      brBc[pattern[pattern_size - 1]][a] = 1;
}


void tvsbs(search_parameters params){
   int i,j =0;
   int BrBc[SIGMA][SIGMA];
   unsigned char firstCh, lastCh;
   TVSBSpreBrBc(params.pattern, params.pattern_size, BrBc);
   firstCh = params.pattern[0];
   lastCh = params.pattern[params.pattern_size -1];
   for (i=0; i<params.pattern_size; i++) params.text[params.text_size+i]=params.text[params.text_size+params.pattern_size+i]=params.pattern[i];
   while(j <= params.text_size - params.pattern_size){
      if (lastCh == params.text[j + params.pattern_size - 1] && firstCh == params.text[j]) {
         for (i = params.pattern_size-2; i > 0 && params.pattern[i] == params.text[j + i]; i--);
         if (i <= 0) params.match[j] = 1;
      }
      j += BrBc[params.text[j + params.pattern_size]][params.text[j+params.pattern_size+1]];
   }
 }
