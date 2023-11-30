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
 * This is an implementation of the Two Sliding Window algorithm
 * in A. Hudaib and R. Al-Khalid and D. Suleiman and M. Itriq and A. Al-Anani.
 * A Fast Pattern Matching Algorithm with Two Sliding Windows (TSW). J. Comput. Sci., vol.4, n.5, pp.393--401, (2008).
 */

#include "include/define.h"
#include "tsw.h"
#include "stdio.h"
#include "stdlib.h"

void preBrBcTSW(unsigned char *pattern, int pattern_size, int brBc[SIGMA][SIGMA]) {
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

void tsw(search_parameters params) {
   int j, brBc_left[SIGMA][SIGMA], brBc_right[SIGMA][SIGMA];
   int i, a,b;
   unsigned char x1[XSIZE];
   for (i=params.pattern_size-1, j=0; i>=0; i--, j++) x1[j]=params.pattern[i];

   /* Preprocessing */
   preBrBcTSW(params.pattern, params.pattern_size, brBc_left);
   preBrBcTSW(x1, params.pattern_size, brBc_right);

   /* Searching */
   j = 0; a = params.text_size-params.pattern_size;
   while (j <= a) {
      for (i=0; i<params.pattern_size && params.pattern[i]==params.text[j+i]; i++);
      if (i>=params.pattern_size && j<=a) params.match[a] = 1;

      for (b=0; b<params.pattern_size && params.pattern[b]==params.text[a+b]; b++);
      if (b>=params.pattern_size && j<a) params.match[j] = 1;

      j += brBc_left[params.text[j + params.pattern_size]][params.text[j + params.pattern_size + 1]];
      a -= brBc_right[params.text[a - 1]][params.text[a - 2]];
   }
}
