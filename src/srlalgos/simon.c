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
 * This is an implementation of the Simon algorithm
 * in I. Simon.
 * String matching algorithms and automata. Proceedings of the 1st South American Workshop on String Processing, pp.151--157, Universidade Federal de Minas Gerais, Brazil, (1993).
 */

#include "simon.h"
#include "include/define.h"

#include "stdio.h"
#include "stdlib.h"
#include "string.h"


static int getTransitionSimon(char *x, int m, int p, List L[], char c) {
   List cell;
 
   if (p < m - 1 && x[p + 1] == c)
      return(p + 1);
   else if (p > -1) {
      cell = L[p];
      while (cell != 0)
         if (x[cell->element] == c)
            return(cell->element);
         else
            cell = cell->next;
      return(-1);
   }
   else
      return(-1);
}
 
 
static void setTransitionSimon(int p, int q, List L[]) {
   List cell;
 
   cell = (List)malloc(sizeof(struct _cell));
   if (cell == 0)
      printf("SIMON/setTransition");
   cell->element = q;
   cell->next = L[p];
   L[p] = cell;
}
 
 
static int preSimon(char *x, int m, List L[]) {
   int i, k, ell;
   List cell;
 
   memset(L, 0, (m - 1)*sizeof(List));
   ell = -1;
   for (i = 1; i < m; ++i) {
      k = ell;
      cell = (ell == -1 ? 0 : L[k]);
      ell = -1;
      if (x[i] == x[k + 1])
         ell = k + 1;
      else
         setTransitionSimon(i - 1, k + 1, L);
      while (cell != 0) {
         k = cell->element;
         if (x[i] == x[k])
            ell = k;
         else
            setTransitionSimon(i - 1, k, L);
         cell = cell->next;
      }
   }
   return(ell);
}


void simon(search_parameters params) {
   int j, ell, state;
   List L[XSIZE];
 
   /* Preprocessing */
   ell = preSimon(params.pattern, params.pattern_size, L);
 
   /* Searching */
   for (state = -1, j = 0; j < params.text_size; ++j) {
      state = getTransitionSimon(params.pattern, params.pattern_size, state, L, params.text[j]);
      if (state >= params.pattern_size - 1) {
         params.match[j - params.pattern_size + 1] = 1;
         state = ell;
      }
   }
}
