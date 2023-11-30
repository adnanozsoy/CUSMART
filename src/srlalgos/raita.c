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
 * This is an implementation of the Raita algorithm
 * in T. Raita.
 * Tuning the Boyer-Moore-Horspool string searching algorithm. Softw. Pract. Exp., vol.22, n.10, pp.879--884, (1992).
 */

#include "raita.h"
#include "include/define.h"
#include <stdio.h>
#include <string.h>

void preBmBcRAITA(unsigned char *pattern, int pattern_size, int bmBc[]) {
  int i;
  for (i = 0; i < SIGMA; ++i)
    bmBc[i] = pattern_size;
  for (i = 0; i < pattern_size - 1; ++i)
    bmBc[pattern[i]] = pattern_size - i - 1;
}

void raita(search_parameters params) {
  int j, bmBc[SIGMA], count;
  unsigned char c, firstCh, *secondCh, middleCh, lastCh;

  /* Preprocessing */
  preBmBcRAITA(params.pattern, params.pattern_size, bmBc);
  firstCh = params.pattern[0];
  secondCh = params.pattern + 1;
  middleCh = params.pattern[params.pattern_size/2];
  lastCh = params.pattern[params.pattern_size - 1];

  /* Searching */
  count = 0;
  j = 0;
  while (j <= params.text_size - params.pattern_size) {
    c = params.text[j + params.pattern_size - 1];
    if (lastCh == c && middleCh == params.text[j + params.pattern_size/2] &&
	firstCh == params.text[j] &&
	memcmp(secondCh, params.text + j + 1, params.pattern_size - 2) == 0)
      params.match[j] = 1;
    j += bmBc[c];
  }
}
