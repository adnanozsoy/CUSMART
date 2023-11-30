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
 * This is an implementation of the Deterministic Finite Automaton algorithm
 * in T. H. Cormen and C. E. Leiserson and R. L. Rivest and C. Stein.
 * Introduction to Algorithms. MIT Press, (2001).
 */

#include "dfa.h"
#include "include/define.h"

#include "stdio.h"
#include "stdlib.h"
#include "string.h"

#define SIGMA 256


static void preSMA(unsigned char *x, int m, int *ttransSMA) {
   int i, j,state, target, oldTarget;
   int c;

   memset(ttransSMA, 0, SIGMA*sizeof(int));
   for (state = 0, i = 0; i < m; ++i) {
      oldTarget = ttransSMA[state*SIGMA + x[i]];
      target = state+1;
      ttransSMA[state*SIGMA + x[i]] = target;
      for (j=0, c=0; j < SIGMA; ++c, ++j)
         ttransSMA[target*SIGMA + c] = ttransSMA[oldTarget*SIGMA + c];
      state = target;
   }
}

void dfa(search_parameters params) {
   int j, count;
   int *ttransSMA;
   unsigned int state;

   /* Preprocessing */
   ttransSMA = (int *)malloc((params.pattern_size+1)*SIGMA*sizeof(int));
   memset(ttransSMA, -1, (params.pattern_size+1)*SIGMA*sizeof(int));
   preSMA(params.pattern, params.pattern_size, ttransSMA);
   /* Searching */
   for (state = 0, j = 0; j < params.text_size; ++j) {
      state = ttransSMA[state*SIGMA + params.text[j]];
      if (state==params.pattern_size) 
         params.match[j - params.pattern_size + 1] = 1;
   }
   free(ttransSMA);
}

