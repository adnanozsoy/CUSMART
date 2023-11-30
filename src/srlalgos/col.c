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
 * This is an implementation of the Colussi algorithm
 * in L. Colussi.
 * Correctness and efficiency of the pattern matching algorithms. Inf. Comput., vol.95, n.2, pp.225--251, (1991).
 */

#include "col.h"
#include "include/define.h"

#include <stdlib.h>
#include <string.h>

int preColussi(unsigned char *x, int m, int h[], int next[],
               int shift[]) {
   int i, k, nd, q, r, s;
   int hmax[XSIZE], kmin[XSIZE], nhd0[XSIZE], rmin[XSIZE];
   
   /* Computation of hmax */
   i = k = 1;
   do {
      while (x[i] == x[i - k])
         i++;
      hmax[k] = i;
      q = k + 1;
      while (hmax[q - k] + k < i) {
         hmax[q] = hmax[q - k] + k;
         q++;
      }
      k = q;
      if (k == i + 1)
         i = k;
   } while (k <= m);
   
   /* Computation of kmin */
   memset(kmin, 0, m*sizeof(int));
   for (i = m; i >= 1; --i)
      if (hmax[i] < m)
         kmin[hmax[i]] = i;
   
   /* Computation of rmin */
   for (i = m - 1; i >= 0; --i) {
      if (hmax[i + 1] == m)
         r = i + 1;
      if (kmin[i] == 0)
         rmin[i] = r;
      else
         rmin[i] = 0;
   }
   
   /* Computation of h */
   s = -1;
   r = m;
   for (i = 0; i < m; ++i)
      if (kmin[i] == 0)
         h[--r] = i;
      else
         h[++s] = i;
   nd = s;
   
   /* Computation of shift */
   for (i = 0; i <= nd; ++i)
      shift[i] = kmin[h[i]];
   for (i = nd + 1; i < m; ++i)
      shift[i] = rmin[h[i]];
   shift[m] = rmin[0];
   
   /* Computation of nhd0 */
   s = 0;
   for (i = 0; i < m; ++i) {
      nhd0[i] = s;
      if (kmin[i] > 0)
         ++s;
   }
   
   
   /* Computation of next */
   for (i = 0; i <= nd; ++i)
      next[i] = nhd0[h[i] - kmin[h[i]]];
   for (i = nd + 1; i < m; ++i)
      next[i] = nhd0[m - rmin[h[i]]];
   next[m] = nhd0[m - rmin[h[m - 1]]];
   
   return(nd);
}

void col(search_parameters params) 
{
   int h[XSIZE], next[XSIZE], shift[XSIZE];
   
   /* Processing */
   int nd = preColussi(params.pattern, params.pattern_size, h, next, shift);
   
   /* Searching */
   int i = 0;
   int j = 0;
   int last = -1;
   while (j <= params.text_size - params.pattern_size) {
      while (i < params.pattern_size && last < j + h[i] && params.pattern[h[i]] == params.text[j + h[i]])
         i++;
      if (i >= params.pattern_size || last >= j + h[i]) {
         params.match[j] = 1;
         i = params.pattern_size;
      }
      if (i > nd)
         last = j + params.pattern_size - 1;
      j += shift[i];
      i = next[i];
   }
}
