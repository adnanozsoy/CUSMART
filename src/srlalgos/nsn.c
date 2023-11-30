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
 * This is an implementation of the Not So Naive algorithm
 * in C. Hancart.
 * Analyse exacte et en moyenne d'algorithmes de recherche d'un motif dans un texte. (1993).
 */

#include "nsn.h"
#include "include/define.h"

#include "stdio.h"
#include "stdlib.h"
#include "string.h"

void nsn(search_parameters params) {
  int j, k, ell, count;
  
  /* Preprocessing */
  if (params.pattern[0] == params.pattern[1]) {
    k = 2;
    ell = 1;
  }
  else {
    k = 1;
    ell = 2;
  }
  count = 0;
  /* Searching */
  j = 0;
  while (j <= params.text_size - params.pattern_size)
    if (params.pattern[1] != params.text[j + 1])
      j += k;
    else {
      if (memcmp(params.pattern + 2, params.text + j + 2, params.pattern_size - 2) == 0 &&
	  params.pattern[0] == params.text[j])
	params.match[j] = 1;
      j += ell;
    }
}
