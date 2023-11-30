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
 * This is an implementation of the KMP Skip Search algorithm
 * in C. Charras and T. Lecroq and J. D. Pehoushek.
 * A Very Fast String Matching Algorithm for Small Alphabets and Long Patterns. 
 * Proceedings of the 9th Annual Symposium on Combinatorial Pattern Matching, Lecture Notes in Computer Science, n.1448, pp.55--64, Springer-Verlag, Berlin, rutgers, (1998).
 */

#include "kmpskip.h"
#include "include/define.h"
#include <stdio.h>
#include <string.h>

void preKmp(unsigned char *pattern, int pattern_size, int kmpNext[]) {
   int i, j;
   i = 0;
   j = kmpNext[0] = -1;
   while (i < pattern_size) {
      while (j > -1 && pattern[i] != pattern[j])
         j = kmpNext[j];
      i++;
      j++;
      if (i<pattern_size && pattern[i] == pattern[j])
         kmpNext[i] = kmpNext[j];
      else
         kmpNext[i] = j;
   }
}

void preMp(char *pattern, int pattern_size, int mpNext[]) {
   int i, j;

   i = 0;
   j = mpNext[0] = -1;
   while (i < pattern_size) {
      while (j > -1 && pattern[i] != pattern[j])
         j = mpNext[j];
      mpNext[++i] = ++j;
   }
}

int attempt(char *text, char *pattern, int pattern_size, int start, int wall) {
   int k;
   k = wall - start;
   while (k < pattern_size && pattern[k] == text[k + start])
      ++k;
   return(k);
}

void kmpskip(search_parameters params) {
   int i, j, k, kmpStart, per, start, wall;
   int kmpNext[XSIZE], list[XSIZE], mpNext[XSIZE], z[SIGMA];

   /* Preprocessing */
   preMp(params.pattern, params.pattern_size, mpNext);
   preKmp(params.pattern, params.pattern_size, kmpNext);
   memset(z, -1, SIGMA*sizeof(int));
   memset(list, -1, params.pattern_size*sizeof(int));
   z[params.pattern[0]] = 0;
   for (i = 1; i < params.pattern_size; ++i) {
      list[i] = z[params.pattern[i]];
      z[params.pattern[i]] = i;
   }

   /* Searching */
   wall = 0;
   per = params.pattern_size - kmpNext[params.pattern_size];
   i = j = -1;
   do {
      j += params.pattern_size;
   } while (j < params.text_size && z[params.text[j]] < 0);
   if (j >= params.text_size){
     return;
   }
   i = z[params.text[j]];
   start = j - i;
   while (start <= params.text_size - params.pattern_size) {
      if (start > wall)
         wall = start;
      k = attempt(params.text, params.pattern, params.pattern_size, start, wall);
      wall = start + k;
      if (k == params.pattern_size) {
         params.match[start] = 1;
         i -= per;
      }
      else
         i = list[i];
      if (i < 0) {
         do {
            j += params.pattern_size;
         } while (j < params.text_size && z[params.text[j]] < 0);
         if (j >= params.text_size)
            return;
         i = z[params.text[j]];
      }
      kmpStart = start + k - kmpNext[k];
      k = kmpNext[k];
      start = j - i;
      while (start < kmpStart ||
             (kmpStart < start && start < wall)) {
         if (start < kmpStart) {
            i = list[i];
            if (i < 0) {
               do {
                  j += params.pattern_size;
               } while (j < params.text_size && z[params.text[j]] < 0);
               if (j >= params.text_size)
                  return;
               i = z[params.text[j]];
            }
            start = j - i;
         }
         else {
            kmpStart += (k - mpNext[k]);
            k = mpNext[k];
         }
      }
   }
}
