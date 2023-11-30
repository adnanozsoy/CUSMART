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
 * This is an implementation of the Backward Nondeterministic DAWG Matching for long patterns algorithm
 * in H. Peltola and J. Tarhio.
 * Alternative Algorithms for Bit-Parallel String Matching.
 * Proceedings of the 10th International Symposium on String Processing and Information Retrieval SPIRE'03, Lecture Notes in Computer Science, vol.2857, pp.80--94, Springer-Verlag, Berlin, Manaus, Brazil, (2003).
 */

#include "lbndm.h"
#include "include/define.h"

void lbndm(search_parameters params)
{
   unsigned int B[SIGMA] = {0};
   unsigned int M;
   int k;
   int i, j, l;
   int m1, m2, rmd;

   /* Preprocessing */
   M = 1 << (WORD-1);
   k = (params.pattern_size-1)/WORD+1;
   m1 = params.pattern_size-1;
   m2 = m1-k;
   rmd = params.pattern_size-(params.pattern_size/k)*k;
   for (i=params.pattern_size/k, l=params.pattern_size; i>0; i--, l-=k)
      for (j=k; j>0; j--)
         B[params.pattern[l-j]] |= 1 << (WORD-i);

   /* Searching */
   j = 0;
   while (j <= params.text_size-params.pattern_size) {
      unsigned int D = B[params.text[j+m1]];
      int last = (D & M) ? params.pattern_size-k-rmd : params.pattern_size-rmd;
      l = m2;
      while (D) {
         D = (D << 1) & B[params.text[j+l]];
         if (D & M) {
            if (l < k+rmd) {
               char *yy = params.text+j;
               for (int jj=0; jj<k; jj++) {
                  if (params.pattern_size+jj > params.text_size-j) break;
                  i = 0;
                  while(i<params.pattern_size && yy[i]==params.pattern[i]) i++;
                  if(i>=params.pattern_size) params.match[j] = 1;
                  yy++;
               }
               break;
            }
            last = l-(k+rmd)+1;
         }
         l -= k;
      }
      j += last;
   }
}
