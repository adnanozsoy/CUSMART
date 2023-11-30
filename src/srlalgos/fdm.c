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
 * This is an implementation of the Forward DAWG Matching algorithm
 * in M. Crochemore and W. Rytter.
 * Text algorithms. Oxford University Press, (1994).
 */

#include "fdm.h"
#include "include/define.h"
#include "include/AUTOMATON.h"

void fdm(search_parameters params) {
   int j, init, ell, state;
   int *ttrans, *tlength, *tsuffix;
   unsigned char *tterminal;
 
   /* Preprocessing */
   ttrans = (int *)malloc(3*params.pattern_size*SIGMA*sizeof(int));
   memset(ttrans, -1, 3*params.pattern_size*SIGMA*sizeof(int));
   tlength = (int *)calloc(3*params.pattern_size, sizeof(int));
   tsuffix = (int *)calloc(3*params.pattern_size, sizeof(int));
   tterminal = (char *)calloc(3*params.pattern_size, sizeof(char));
   buildSimpleSuffixAutomaton(params.pattern, params.pattern_size, ttrans, tlength, tsuffix, tterminal);
   init = 0;
 
   /* Searching */
   ell = 0;
   state = init;
   for (j = 0; j < params.text_size; ++j) {
      if (getTarget(state, params.text[j]) != UNDEFINED) {
         ++ell;
         state = getTarget(state, params.text[j]);
      }
      else {
         while (state != init && getTarget(state, params.text[j]) == UNDEFINED)
            state = getSuffixLink(state);
         if (getTarget(state, params.text[j]) != UNDEFINED) {
            ell = getLength(state) + 1;
            state = getTarget(state, params.text[j]);
         }
         else {
            ell = 0;
            state = init;
         }
      }
      if (ell == params.pattern_size)
         params.match[j - params.pattern_size + 1] = 1;
   }
   free(ttrans);
   free(tlength);
   free(tsuffix);
   free(tterminal);
}



