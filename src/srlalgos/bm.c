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
 * This is an implementation of the Boyer Moore algorithm
 * in R. S. Boyer and J. S. Moore.
 * A fast string searching algorithm. Commun. ACM, vol.20, n.10, pp.762--772, (1977).
 */

#include "bm.h"
#include "include/define.h"

#include "stdio.h"
#include "stdlib.h"
#include "string.h"


static void preBmBc(unsigned char *x, int m, int bmBc[]) {
   int i;
   for (i = 0; i < SIGMA; ++i)
      bmBc[i] = m;
   for (i = 0; i < m - 1; ++i)
      bmBc[x[i]] = m - i - 1;
}

static void suffixes(unsigned char *x, int m, int *suff) {
   int f, g, i;
   suff[m - 1] = m;
   g = m - 1;
   for (i = m - 2; i >= 0; --i) {
      if (i > g && suff[i + m - 1 - f] < i - g)
         suff[i] = suff[i + m - 1 - f];
      else {
         if (i < g)
            g = i;
         f = i;
         while (g >= 0 && x[g] == x[g + m - 1 - f])
            --g;
         suff[i] = f - g;
      }
   }
}

static void preBmGs(unsigned char *x, int m, int bmGs[]) {
   int i, j, suff[XSIZE];
   suffixes(x, m, suff);
   for (i = 0; i < m; ++i) bmGs[i] = m;
   j = 0;
   for (i = m - 1; i >= 0; --i)
      if (suff[i] == i + 1)
         for (; j < m - 1 - i; ++j)
            if (bmGs[j] == m)
               bmGs[j] = m - 1 - i;
   for (i = 0; i <= m - 2; ++i)
      bmGs[m - 1 - suff[i]] = m - 1 - i;
}

void bm(search_parameters params) {
   int i, j, bmGs[XSIZE], bmBc[SIGMA];
 
   /* Preprocessing */
   preBmGs(params.pattern, params.pattern_size, bmGs);
   preBmBc(params.pattern, params.pattern_size, bmBc);
 
   /* Searching */
   j = 0;
   while (j <= params.text_size - params.pattern_size) {
      for ( i = params.pattern_size - 1; i >= 0 && 
            params.pattern[i] == params.text[i + j]; --i);
      if (i < 0) {
         params.match[j] = 1;
         j += bmGs[0];
      }
      else
         j += MAX(bmGs[i], bmBc[params.text[i + j]] - params.pattern_size + 1 + i);
   }
}

