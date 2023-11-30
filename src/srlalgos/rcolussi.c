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
 * This is an implementation of the Reverse Colussi algorithm
 * in L. Colussi.
 * Fastest pattern matching in strings. J. Algorithms, vol.16, n.2, pp.163--189, (1994).
 */

#include "rcolussi.h"
#include "include/define.h"
#include <stdlib.h>
#include <string.h>

static void preRc(char *x, int m, int h[], int rcBc[SIGMA][XSIZE], int rcGs[]) {
   int a, i, j, k, q, r, s,
   hmin[XSIZE], kmin[XSIZE], link[XSIZE],
   locc[SIGMA], rmin[XSIZE];
 
   for (a = 0; a < SIGMA; ++a)
      locc[a] = -1;
   link[0] = -1;
   for (i = 0; i < m - 1; ++i) {
      link[i + 1] = locc[x[i]];
      locc[x[i]] = i;
   }

   for (a = 0; a < SIGMA; ++a)
      for (s = 1; s <= m; ++s) {
         i = locc[a];
         j = link[m - s];
         while (i - j != s && j >= 0)
            if (i - j > s)
               i = link[i + 1];
            else
               j = link[j + 1];
         while (i - j > s)
            i = link[i + 1];
         rcBc[a][s] = m - i - 1;
      }
 
   k = 1;
   i = m - 1;
   while (k <= m) {
      while (i - k >= 0 && x[i - k] == x[i])
         --i;
      hmin[k] = i;
      q = k + 1;
      while (hmin[q - k] - (q - k) > i) {
         hmin[q] = hmin[q - k];
         ++q;
      }
      i += (q - k);
      k = q;
      if (i == m)
         i = m - 1;
   }
 
   memset(kmin, 0, m * sizeof(int));
   for (k = m; k > 0; --k)
      kmin[hmin[k]] = k;
 
   for (i = m - 1; i >= 0; --i) {
      if (hmin[i + 1] == i)
         r = i + 1;
      rmin[i] = r;
   }
 
   i = 1;
   for (k = 1; k <= m; ++k)
      if (hmin[k] != m - 1 && kmin[hmin[k]] == k) {
         h[i] = hmin[k];
         rcGs[i++] = k;
      }
   i = m-1;
   for (j = m - 2; j >= 0; --j)
      if (kmin[j] == 0) {
         h[i] = j;
         rcGs[i--] = rmin[j];
      }
   rcGs[m] = rmin[0];
}
 
 
void rcolussi(search_parameters params) {
   int i, j, s, rcBc[SIGMA][XSIZE], rcGs[XSIZE], h[XSIZE], count;
   /* Preprocessing */
   preRc(params.pattern, params.pattern_size, h, rcBc, rcGs);
 
   /* Searching */
   count = 0;
   j = 0;
   s = params.pattern_size;
   while (j <= params.text_size - params.pattern_size) {
      while (j <= params.text_size - params.pattern_size &&
         params.pattern[params.pattern_size - 1] != params.text[j + params.pattern_size - 1]) {
         s = rcBc[params.text[j + params.pattern_size - 1]][s];
         j += s;
      }
      for (i = 1; i < params.pattern_size && params.pattern[h[i]] == params.text[j + h[i]]; ++i);
      if (i >= params.pattern_size && j <= params.text_size - params.pattern_size)
         params.match[j] = 1;
      s = rcGs[i];
      j += s;
   }
}


