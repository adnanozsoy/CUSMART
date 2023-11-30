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
 * This is an implementation of the Factorized Shift And algorithm
 * in D. Cantone and S. Faro and E. Giaquinta.
 * A Compact Representation of Nondeterministic (Suffix) Automata for the Bit-Parallel Approach. 
 * Combinatorial Pattern Matching, Lecture Notes in Computer Science, vol.6129, pp.288--298, Springer-Verlag, Berlin, (2010).
 */

#include "ksa.h"
#include "include/define.h"

#define CHAR_BIT 8
#define WORD_TYPE unsigned int
#define WORD_BITS (sizeof(WORD_TYPE)*CHAR_BIT)
#include "stdlib.h"
#include "string.h"

void ksa(search_parameters params){
   int i, j, k, m1;
   int beg, end;
   WORD_TYPE D, D_, M;
   WORD_TYPE B[SIGMA][SIGMA] = {{0}};
   WORD_TYPE L[SIGMA] = {0};
   unsigned char c;

   /* Preprocessing */
   end = 1;
   for (k = 1; k < WORD_BITS-1; k++) {
      char occ[SIGMA] = {0};
      while (end < params.pattern_size && occ[params.pattern[end]] == 0) {
         occ[params.pattern[end]] = 1;
         end++;
      }
   }
   m1 = end;
   k = 1;
   beg = 1;
   end = 1;
   B[params.pattern[0]][params.pattern[1]] = 1;
   L[params.pattern[0]] = 1;
   for (;;) {
      char occ[SIGMA] = {0};
      while (end < m1 && occ[params.pattern[end]] == 0) {
         occ[params.pattern[end]] = 1;
         end++;
      }
      for (i = beg+1; i < end; i++)
         B[params.pattern[i-1]][params.pattern[i]] |= 1 << k;
      if (end < m1) {
         B[params.pattern[end-1]][params.pattern[end]] |= 1 << k;
         L[params.pattern[end-1]] |= 1 << k;
      } else {
         M = 1 << k;
         if (end > beg+1) {
            L[params.pattern[end-2]] |= 1L << k;
            M <<= 1;
         }
         break;
      }
      beg = end;
      k++;
   }

   /* Searching */
   D = 0;
   c = params.text[0];
   for (j = 1; j < params.text_size; j++) {
      D = (D|1) & B[c][params.text[j]];
      D_ = D & L[c];
      D += D_;
      c = params.text[j];
      if (D & M) {
         if (!strncmp(params.pattern+m1, params.text+j+1, params.pattern_size-m1)) {
            params.match[j] = 1;
         }
      }
   }
}
