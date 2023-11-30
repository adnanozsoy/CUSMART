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
 * This is an implementation of the Maximal Shift algorithm
 * in D. M. Sunday.
 * A very fast substring search algorithm. Commun. ACM, vol.33, n.8, pp.132--142, (1990).
 */

#include "ms.h"
#include "include/define.h"
#include <stdio.h>
#include <stdlib.h>

typedef struct patternScanOrder {
  int loc;
  char c;
} pattern;

static int minShift[XSIZE];

void computeMinShift(unsigned char *x, int m) {
  int i, j;
  for (i = 0; i < m; ++i) {
    for (j = i - 1; j >= 0; --j)
      if (x[i] == x[j])
	break;
    minShift[i] = i - j;
  }
}

void orderPatternMS(unsigned char *x, int m, int (*pcmp)(), pattern *pat) {
  int i;
  for (i = 0; i < m; ++i) {
    pat[i].loc = i;
    pat[i].c = x[i];
  }
  qsort(pat, m, sizeof(pattern), pcmp);
}

void preQsBcMS(unsigned char *x, int m, int qbc[]) {
  int i;
  for (i=0;i<SIGMA;i++)   qbc[i]=m+1;
  for (i=0;i<m;i++) qbc[x[i]]=m-i;
}

int matchShiftMS(unsigned char *x, int m, int ploc, int lshift, pattern *pat) {
  int i, j;
  for (; lshift < m; ++lshift) {
    i = ploc;
    while (--i >= 0) {
      if ((j = (pat[i].loc - lshift)) < 0)  continue;
      if (pat[i].c != x[j]) break;
    }
    if (i < 0) break;
  }
  return(lshift);
}

void preAdaptedGsMS(unsigned char *x, int m, int adaptedGs[], pattern *pat) {
  int lshift, i, ploc;

  adaptedGs[0] = lshift = 1;
  for (ploc = 1; ploc <= m; ++ploc) {
    lshift = matchShiftMS(x, m, ploc, lshift, pat);
    adaptedGs[ploc] = lshift;
  }
  for (ploc = 0; ploc <= m; ++ploc) {
    lshift = adaptedGs[ploc];
    while (lshift < m) {
      i = pat[ploc].loc - lshift;
      if (i < 0 || pat[ploc].c != x[i])
	break;
      ++lshift;
      lshift = matchShiftMS(x, m, ploc, lshift, pat);
    }
    adaptedGs[ploc] = lshift;
  }
}


int maxShiftPcmp(pattern *pat1, pattern *pat2) {
  int dsh;
  dsh = minShift[pat2->loc] - minShift[pat1->loc];
  return(dsh ? dsh : (pat2->loc - pat1->loc));
}

void ms(search_parameters params) {
  int i, j, qsBc[SIGMA], adaptedGs[XSIZE], count;
  pattern pat[XSIZE];

  /* Preprocessing */
  computeMinShift(params.pattern ,params.pattern_size);
  orderPatternMS(params.pattern, params.pattern_size, maxShiftPcmp, pat);
  preQsBcMS(params.pattern, params.pattern_size, qsBc);
  preAdaptedGsMS(params.pattern, params.pattern_size, adaptedGs, pat);

  /* Searching */
  count = 0;
  j = 0;
  while (j <= params.text_size - params.pattern_size) {
    i = 0;
    while (i < params.pattern_size && pat[i].c == params.text[j + pat[i].loc])  ++i;
    if (i >= params.pattern_size)
      params.match[j] = 1;
    j += MAX(adaptedGs[i], qsBc[params.text[j + params.pattern_size]]);
  }
}
