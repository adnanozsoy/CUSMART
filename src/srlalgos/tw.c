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
 * This is an implementation of the Two Way algorithm
 * in M. Crochemore and D. Perrin.
 * Two-way string-matching. J. Assoc. Comput. Mach., vol.38, n.3, pp.651--675, (1991).
 */

#include "tw.h"
#include "include/define.h"

#include "stdio.h"
#include "stdlib.h"
#include "string.h"



/* Computing of the maximal suffix for <= */
int maxSuf(char *x, int m, int *p) {
   int ms, j, k;
   char a, b;

   ms = -1;
   j = 0;
   k = *p = 1;
   while (j + k < m) {
      a = x[j + k];
      b = x[ms + k];
      if (a < b) {
         j += k;
         k = 1;
         *p = j - ms;
      }
      else
         if (a == b)
            if (k != *p)
               ++k;
            else {
               j += *p;
               k = 1;
            }
         else { /* a > b */
            ms = j;
            j = ms + 1;
            k = *p = 1;
         }
   }
   return(ms);
}
 
/* Computing of the maximal suffix for >= */
int maxSufTilde(char *x, int m, int *p) {
   int ms, j, k;
   char a, b;

   ms = -1;
   j = 0;
   k = *p = 1;
   while (j + k < m) {
      a = x[j + k];
      b = x[ms + k];
      if (a > b) {
         j += k;
         k = 1;
         *p = j - ms;
      }
      else
         if (a == b)
            if (k != *p)
               ++k;
            else {
               j += *p;
               k = 1;
            }
         else { /* a < b */
            ms = j;
            j = ms + 1;
            k = *p = 1;
         }
   }
   return(ms);
}

void tw(search_parameters params) {
   int i, j, ell, memory, p, per, q;

   /* Preprocessing */
   i = maxSuf(params.pattern, params.pattern_size, &p);
   j = maxSufTilde(params.pattern, params.pattern_size, &q);
   if (i > j) {
      ell = i;
      per = p;
   }
   else {
      ell = j;
      per = q;
   }

   /* Searching */
   
   if (memcmp(params.pattern, params.pattern + per, ell + 1) == 0) {
      j = 0;
      memory = -1;
      while (j <= params.text_size - params.pattern_size) {
         i = MAX(ell, memory) + 1;
         while (i < params.pattern_size && params.pattern[i] == params.text[i + j])
            ++i;
         if (i >= params.pattern_size) {
            i = ell;
            while (i > memory && params.pattern[i] == params.text[i + j])
               --i;
            if (i <= memory)
               params.match[j]=1;
            j += per;
            memory = params.pattern_size - per - 1;
         }
         else {
            j += (i - ell);
            memory = -1;
         }
      }
   }
   else {
      per = MAX(ell + 1, params.pattern_size - ell - 1) + 1;
      j = 0;
      while (j <= params.text_size - params.pattern_size) {
         i = ell + 1;
         while (i < params.pattern_size && params.pattern[i] == params.text[i + j])
            ++i;
         if (i >= params.pattern_size) {
            i = ell;
            while (i >= 0 && params.pattern[i] == params.text[i + j])
               --i;
            if (i < 0)
               params.match[j]=1;
            j += per;
         }
         else
            j += (i - ell);
      }
   }
   
}
