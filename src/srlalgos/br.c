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
 * This is an implementation of the Berry Ravindran algorithm
 * in T. Berry and S. Ravindran.
 * A fast string matching algorithm and experimental results. Proceedings of the Prague Stringology Club Workshop '99, pp.16--28, ctu, (1999).
 */

#include "br.h"
#include "include/define.h"


void preBrBc(unsigned char *x, int m, int brBc[SIGMA][SIGMA]) { 
   int a, b, i; 
   for (a = 0; a < SIGMA; ++a) 
      for (b = 0; b < SIGMA; ++b) 
         brBc[a][b] = m + 2; 
   for (a = 0; a < SIGMA; ++a) 
      brBc[a][x[0]] = m + 1; 
   for (i = 0; i < m - 1; ++i) 
      brBc[x[i]][x[i + 1]] = m - i; 
   for (a = 0; a < SIGMA; ++a) 
      brBc[x[m - 1]][a] = 1; 
} 


void br(search_parameters params) { 
   int j, brBc[SIGMA][SIGMA]; 
   int i; 

   /* Preprocessing */ 
   preBrBc(params.pattern, params.pattern_size, brBc); 

   /* Searching */ 
   params.text[params.text_size + 1] = '\0'; 
   j = 0; 
   while (j <= params.text_size - params.pattern_size) { 
      for (i=0; i<params.pattern_size && params.pattern[i]==params.text[j+i]; i++); 
      if (i>=params.pattern_size) params.match[j]=1; 
      j += brBc[params.text[j + params.pattern_size]][params.text[j + params.pattern_size + 1]]; 
   } 
} 



