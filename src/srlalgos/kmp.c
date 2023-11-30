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
 * This is an implementation of the Knuth Morris-Pratt algorithm
 * in D. E. Knuth and J. H. Morris and V. R. Pratt.
 * Fast pattern matching in strings. SIAM J. Comput., vol.6, n.1, pp.323--350, (1977).
 */

#include "kmp.h"
#include "include/define.h"

#include "stdio.h"
#include "stdlib.h"
#include "string.h"


static void preKmp(unsigned char *x, int m, int kmpNext[]) {
   int i, j;
   i = 0;
   j = kmpNext[0] = -1;
   while (i < m) {
      while (j > -1 && x[i] != x[j])
         j = kmpNext[j];
      i++;
      j++;
      if (i<m && x[i] == x[j])
         kmpNext[i] = kmpNext[j];
      else
         kmpNext[i] = j;
   }
}


void kmp(search_parameters params) {
   int i, j, kmpNext[XSIZE], count;

   /* Preprocessing */
   preKmp(params.pattern, params.pattern_size, kmpNext);

   /* Searching */
   i = j = 0;
   while (j < params.text_size) {
      while (i > -1 && params.pattern[i] != params.text[j])
         i = kmpNext[i];
      i++;
      j++;
      if (i >= params.pattern_size) {
         params.match[j - i] = 1;
         i = kmpNext[i];
      }
   }
}


