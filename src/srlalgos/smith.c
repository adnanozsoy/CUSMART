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
 * This is an implementation of the Smith algorithm
 * in P. D. Smith.
 * Experiments with a very fast substring search algorithm. Softw. Pract. Exp., vol.21, n.10, pp.1065--1074, (1991).
 */

#include "smith.h"
#include "include/define.h"
#include <stdio.h>
#include <string.h>

void preQsBcSMITH(unsigned char *P, int m, int qbc[])
{
  int i;
  for (i=0;i<SIGMA;i++)qbc[i]=m+1;
  for (i=0;i<m;i++) qbc[P[i]]=m-i;
}

void preBmBcSMITH(unsigned char *x, int m, int bmBc[]) {
  int i;

  for (i = 0; i < SIGMA; ++i)
    bmBc[i] = m;
  for (i = 0; i < m - 1; ++i)
    bmBc[x[i]] = m - i - 1;
}

void smith(search_parameters params) {
  int j, bmBc[SIGMA], qsBc[SIGMA];

  /* Preprocessing */
  preBmBcSMITH(params.pattern, params.pattern_size, bmBc);
  preQsBcSMITH(params.pattern, params.pattern_size, qsBc);

  /* Searching */
  j = 0;
  while (j<= params.text_size - params.pattern_size) {
    if (memcmp(params.pattern, params.text + j, params.pattern_size) == 0)
      params.match[j] = 1;
    j += MAX(bmBc[params.text[j + params.pattern_size - 1]], qsBc[params.text[j + params.pattern_size]]);
  }
}
