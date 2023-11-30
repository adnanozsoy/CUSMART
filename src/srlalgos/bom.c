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
 * This is an implementation of the Backward Oracle Matching algorithm
 * in C. Allauzen and M. Crochemore and M. Raffinot.
 * Factor oracle: a new structure for pattern matching. 
 * SOFSEM'99, Theory and Practice of Informatics, Lecture Notes in Computer Science, n.1725, pp.291--306, Springer-Verlag, Berlin, Milovy, Czech Republic, (1999).
 */

#include "bom.h"
#include "include/define.h"
#include <string.h>
#include <stdlib.h>

int getTransition(unsigned char *x, int p, List L[], unsigned char c) {
   List cell;
   if (p > 0 && x[p - 1] == c) return(p - 1);
   else {
      cell = L[p];
      while (cell != NULL)
         if (x[cell->element] == c)
            return(cell->element);
         else
            cell = cell->next;
      return(UNDEFINED);
   }
}

void setTransition(int p, int q, List L[]) {
   List cell;
   cell = (List)malloc(sizeof(struct _cell));
   cell->element = q;
   cell->next = L[p];
   L[p] = cell;
}

void oracle(unsigned char *x, int m, char T[], List L[]) {
   int i, p, q;
   int S[XSIZE + 1];
   char c;
   S[m] = m + 1;
   for (i = m; i > 0; --i) {
      c = x[i - 1];
      p = S[i];
      while (p <= m &&
             (q = getTransition(x, p, L, c)) ==
             UNDEFINED) {
         setTransition(p, i - 1, L);
         p = S[p];
      }
      S[i - 1] = (p == m + 1 ? m : q);
   }
   p = 0;
   while (p <= m) {
      T[p] = TRUE;
      p = S[p];
   }
}


void bom(search_parameters params) {
   char T[XSIZE + 1];
   List L[XSIZE + 1];
   int i, j, p, period, q, shift;

   /* Preprocessing */
   memset(L, 0, (params.pattern_size + 1)*sizeof(List));
   memset(T, FALSE, (params.pattern_size + 1)*sizeof(char));
   oracle(params.pattern, params.pattern_size, T, L);

   /* Searching */

   j = 0;
   while (j <= params.text_size - params.pattern_size) {
      i = params.pattern_size - 1;
      p = params.pattern_size;
      shift = params.pattern_size;
      while (i + j >= 0 && (q = getTransition(params.pattern, p, L, params.text[i + j])) != UNDEFINED) {
         p = q;
         if (T[p] == TRUE) {
            period = shift;
            shift = i;
         }
         --i;
      }
      if (i < 0) {
         params.match[j] = 1;
         shift = period;
      }
      j += shift;
   }
   for (i=0; i<=params.pattern_size; i++) free(L[i]);
}
