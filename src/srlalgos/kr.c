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
 * This is an implementation of the Karp Rabin algorithm
 * in R. M. Karp and M. O. Rabin.
 * Efficient randomized pattern-matching algorithms. ibmjrd, vol.31, n.2, pp.249--260, (1987).
 */

#include "kr.h"
#include "include/define.h"

#include "stdio.h"
#include "stdlib.h"
#include "string.h"

#define REHASH(a, b, h) ((((h) - (a)*d) << 1) + (b))

void kr(search_parameters params) {
	int d, hpattern, htext, i, j;

	/* Preprocessing */
	for (d = i = 1; i < params.pattern_size; ++i)
		d = (d<<1);

	for (htext = hpattern = i = 0; i < params.pattern_size; ++i) {
		hpattern = ((hpattern<<1) + params.pattern[i]);
		htext = ((htext<<1) + params.text[i]);
	}

	/* Searching */
	j = 0;
	while (j <= params.text_size - params.pattern_size) {
		if (hpattern == htext && memcmp(params.pattern, params.text + j, params.pattern_size) == 0) 
			params.match[j] = 1;
		
		htext = REHASH(params.text[j], params.text[j + params.pattern_size], htext);
		++j;
	}
}

