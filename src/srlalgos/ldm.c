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
 * This is an implementation of the Linear DAWG Matching algorithm
 * in L. He and B. Fang and J. Sui.
 * The wide window string matching algorithm. Theor. Comput. Sci., vol.332, n.1-3, pp.391--404, Elsevier Science Publishers Ltd., Essex, UK, (2005).
 */

#include "ldm.h"
#include "include/AUTOMATON.h"

void ldm(search_parameters params) {
   int k, R, L, r, ell, end;
   int *ttrans, *tlength, *tsuffix;
   int *ttransSMA;
   char *tterminal;
   unsigned char *xR;
 
   /* Preprocessing */
   xR = reverse(params.pattern, params.pattern_size);
   ttrans = (int *)malloc(3*params.pattern_size*SIGMA*sizeof(int));
   memset(ttrans, -1, 3*params.pattern_size*SIGMA*sizeof(int));
   tlength = (int *)calloc(3*params.pattern_size, sizeof(int));
   tsuffix = (int *)calloc(3*params.pattern_size, sizeof(int));
   tterminal = (char *)calloc(3*params.pattern_size, sizeof(char));
   buildSimpleSuffixAutomaton(xR, params.pattern_size, ttrans, tlength, tsuffix, tterminal);

   ttransSMA = (int *)malloc((params.pattern_size+1)*SIGMA*sizeof(int));
   memset(ttransSMA, -1, (params.pattern_size+1)*SIGMA*sizeof(int));
   preSMA(params.pattern, params.pattern_size, ttransSMA);
   end = params.text_size/params.pattern_size;
   if (params.text_size%params.pattern_size > 0) ++end;

   /* Searching */
   for (k = 1; k < end; ++k) {
      R = L = r = ell = 0;
      while (L != UNDEFINED && k*params.pattern_size-1-ell >= 0) {
         L = getTarget(L, params.text[k*params.pattern_size-1-ell]);
         ++ell;
         if (L != UNDEFINED && isTerminal(L))
            R = ell;
      }
      while (R > r-k*params.pattern_size) {
         if (R == params.pattern_size) params.match[k*params.pattern_size+r-params.pattern_size] = 1;
         ++r;
         if (r == params.pattern_size) break;
         R = getSMA(R, params.text[k*params.pattern_size-1+r]);
      }
   }
   for (k = (end-1)*params.pattern_size; k <= params.text_size - params.pattern_size; ++k) {
      for (r = 0; r < params.pattern_size && params.pattern[r] == params.text[r + k]; ++r);
      if (r >= params.pattern_size) params.match[k] = 1;
   }
   free(ttrans);
   free(tlength);
   free(tsuffix);
   free(ttransSMA);
   free(tterminal);
}
