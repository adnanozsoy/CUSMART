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
 * This is an implementation of the Apostolico Giancarlo algorithm
 * in A. Apostolico and R. Giancarlo. 
 * The Boyer-Moore-Galil string searching strategies revisited. SIAM J. Comput., vol.15, n.1, pp.98--105, (1986).
 */

#include "include/define.h"
#include "ag.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"

static void preBmBc(unsigned char *x, int m, int bmBc[]) {
    int i;
    for (i = 0; i < SIGMA; ++i) bmBc[i] = m;
    for (i = 0; i < m - 1; ++i) bmBc[x[i]] = m - i - 1;
}
 
 
static void suffixes(unsigned char *x, int m, int *suff) {
    int f = 0, g, i;
    suff[m - 1] = m;
    g = m - 1;
    for (i = m - 2; i >= 0; --i) {
        if (i > g && suff[i + m - 1 - f] < i - g)
            suff[i] = suff[i + m - 1 - f];
        else {
            if (i < g) g = i;
            f = i;
            while (g >= 0 && x[g] == x[g + m - 1 - f]) --g;
            suff[i] = f - g;
        }
    }
}

static void preBmGsAG(unsigned char *x, int m, int bmGs[], int suff[]) {
    int i, j;
    suffixes(x, m, suff);
    for (i = 0; i < m; ++i) bmGs[i] = m;
    j = 0;
    for (i = m - 1; i >= 0; --i)
        if (suff[i] == i + 1)
            for (; j < m - 1 - i; ++j)
                if (bmGs[j] == m)
                   bmGs[j] = m - 1 - i;
    for (i = 0; i <= m - 2; ++i)
        bmGs[m - 1 - suff[i]] = m - 1 - i;
}



void ag(search_parameters params) {
    int i, j, k, s, shift, count;
    int bmGs[XSIZE], skip[XSIZE], suff[XSIZE], bmBc[SIGMA];
  
    /* Preprocessing */
    preBmGsAG(params.pattern, params.pattern_size, bmGs, suff);
    preBmBc(params.pattern, params.pattern_size, bmBc);
    memset(skip, 0, params.pattern_size*sizeof(int));
  
    /* Searching */
    count = 0;
    j = 0;
   	while (j <= params.text_size - params.pattern_size) {
        i = params.pattern_size - 1;
        while (i >= 0) {
            k = skip[i];
            s = suff[i];
            if (k > 0)
                if (k > s) {
                   if (i + 1 == s)
                      i = (-1);
                   else i -= s;
                   break;
                }
                else {
                    i -= k;
                    if (k < s) break;
                }
            else {
                if (params.pattern[i] == params.text[i + j]) --i;
                else break;
            }
        }
        if (i < 0) {
            params.match[j] = 1;
            skip[params.pattern_size - 1] = params.pattern_size;
            shift = bmGs[0];
        }
        else {
            skip[params.pattern_size - 1] = params.pattern_size - 1 - i;
            shift = MAX(bmGs[i], bmBc[params.text[i + j]] - params.pattern_size + 1 + i);
        }
        j += shift;
        memcpy(skip, skip + shift, (params.pattern_size - shift)*sizeof(int));
        memset(skip + params.pattern_size - shift, 0, shift*sizeof(int));
    }
}

