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
 * This is an implementation of the Tuned Boyer Moore algorithm
 * in A. Hume and D. M. Sunday.
 * Fast string searching. Softw. Pract. Exp., vol.21, n.11, pp.1221--1248, (1991).
 */

#include "tunedbm.h"
#include "include/define.h"

#include "stdio.h"
#include "stdlib.h"
#include "string.h"



void preBmBc(unsigned char *x, int m, int bmBc[]) {
	int i;
	for (i = 0; i < SIGMA; ++i)
		bmBc[i] = m;
	for (i = 0; i < m - 1; ++i)
		bmBc[x[i]] = m - i - 1;
}


void tuned_bm(search_parameters params) {
	int j, k, shift, bmBc[SIGMA];

	/* Preprocessing */
	preBmBc(params.pattern, params.pattern_size, bmBc);
	shift = bmBc[params.pattern[params.pattern_size - 1]];
	bmBc[params.pattern[params.pattern_size - 1]] = 0;
	memset(params.text + params.text_size, params.pattern[params.pattern_size - 1], params.pattern_size);

	/* Searching */
	j = 0;
	while (j <= params.text_size-params.pattern_size) {
		k = bmBc[params.text[j + params.pattern_size -1]];
		while (k !=  0) {
			 j += k; k = bmBc[params.text[j + params.pattern_size -1]];
		}
		if (memcmp(params.pattern, params.text + j, params.pattern_size - 1) == 0 && j <= params.text_size-params.pattern_size)
			 if (j <= params.text_size-params.pattern_size) params.match[j] = 1;
		j += shift;                          
	}
}

