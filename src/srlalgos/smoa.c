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
 * This is an implementation of the String Matching on Ordered Alphabet algorithm
 * in M. Crochemore.
 * String-matching on ordered alphabets. Theor. Comput. Sci., vol.92, n.1, pp.33--47, (1992).
 */

#include "smoa.h"
#include "include/define.h"

#include "stdio.h"
#include "stdlib.h"
#include "string.h"


/* Compute the next maximal suffix. */
void nextMaximalSuffix(char *x, int m,
                       int *i, int *j, int *k, int *p) {
   char a, b;
 
   while (*j + *k < m) {
      a = x[*i + *k];
      b = x[*j + *k];
      if (a == b)
         if (*k == *p) {
            (*j) += *p;
            *k = 1;
         }
         else
            ++(*k);
      else
         if (a > b) {
            (*j) += *k;
            *k = 1;
            *p = *j - *i;
         }
         else {
            *i = *j;
            ++(*j);
            *k = *p = 1;
         }
   }
}
 
 
/* String matching on ordered alphabets algorithm. */
void smoa(search_parameters params) {
   int i, ip, j, jp, k, p, count;
 
   
   /* Searching */
   ip = -1;
   i = j = jp = 0;
   k = p = 1;
   while (j <= params.text_size - params.pattern_size) {
      while (i + j < params.text_size && i < params.pattern_size && params.pattern[i] == params.text[i + j])
         ++i;
      if (i == 0) {
         ++j;
         ip = -1;
         jp = 0;
         k = p = 1;
      }
      else {
         if (i >= params.pattern_size)
            params.match[j]=1;
         nextMaximalSuffix(params.text + j, i+1, &ip, &jp, &k, &p);
         if (ip < 0 ||
             (ip < p &&
              memcmp(params.text + j, params.text + j + p, ip + 1) == 0)) {
            j += p;
            i -= p;
            if (i < 0)
               i = 0;
            if (jp - ip > p)
               jp -= p;
            else {
               ip = -1;
               jp = 0;
               k = p = 1;
            }
         }
         else {
            j += (MAX(ip + 1,
                      MIN(i - ip - 1, jp + 1)) + 1);
            i = jp = 0;
            ip = -1;
            k = p = 1;
         }
      }
   }
   
}

