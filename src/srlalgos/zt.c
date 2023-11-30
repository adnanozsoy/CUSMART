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
 * This is an implementation of the Zhu Takaoka algorithm
 * in R. F. Zhu and T. Takaoka.
 * On improving the average case of the Boyer-Moore string matching algorithm. J. Inform. Process., vol.10, n.3, pp.173--177, (1987).
 */

#include "zt.h"
#include "include/define.h"

#include "stdio.h"
#include "stdlib.h"
#include "string.h"

void suffixes(unsigned char *x, int m, int *suff) {
  int f, g, i;

  suff[m - 1] = m;
  g = m - 1;
  for (i = m - 2; i >= 0; --i) {
    if (i > g && suff[i + m - 1 - f] < i - g)
      suff[i] = suff[i + m - 1 - f];
    else {
      if (i < g)
	g = i;
      f = i;
      while (g >= 0 && x[g] == x[g + m - 1 - f])
	--g;
      suff[i] = f - g;
    }
  }
}

void preBmGs(unsigned char *x, int m, int bmGs[]) {
  int i, j, suff[XSIZE];

  suffixes(x, m, suff);

  for (i = 0; i < m; ++i)
    bmGs[i] = m;
  j = 0;
  for (i = m - 1; i >= 0; --i)
    if (suff[i] == i + 1)
      for (; j < m - 1 - i; ++j)
	if (bmGs[j] == m)
	  bmGs[j] = m - 1 - i;
  for (i = 0; i <= m - 2; ++i)
    bmGs[m - 1 - suff[i]] = m - 1 - i;
}

void preZtBc(unsigned char *x, int m, int ztBc[SIGMA][SIGMA]) {
  int i, j;

  for (i = 0; i < SIGMA; ++i)
    for (j = 0; j < SIGMA; ++j)
      ztBc[i][j] = m;
  for (i = 0; i < SIGMA; ++i)
    ztBc[i][x[0]] = m - 1;
  for (i = 1; i < m - 1; ++i)
    ztBc[x[i - 1]][x[i]] = m - 1 - i;
}

void zt(search_parameters params) {
  int i, j, ztBc[SIGMA][SIGMA], bmGs[XSIZE];

  /* Preprocessing */
  preZtBc(params.pattern, params.pattern_size, ztBc);
  preBmGs(params.pattern, params.pattern_size, bmGs);
  for (i=0; i < params.pattern_size; i++) {
    params.text[params.text_size + i] = params.text[params.text_size + params.pattern_size + i] = params.pattern[i];
  }
    
  /* Searching */
  j = 0;
  while (j <= params.text_size - params.pattern_size) {
    i = params.pattern_size - 1;
    while (i >=0 && params.pattern[i] == params.text[i + j])
      --i;
    if (i < 0) {
      params.match[j] = 1;
      j += bmGs[0];
    }
    else
      j += MAX(bmGs[i], ztBc[params.text[j + params.pattern_size - 2]][params.text[j + params.pattern_size - 1]]);
  }
}
