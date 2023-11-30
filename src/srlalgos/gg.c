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
 * This is an implementation of the Galil Giancarlo algorithm
 * in Z. Galil and R. Giancarlo.
 * On the exact complexity of string matching: upper bounds. SIAM J. Comput., vol.21, n.3, pp.407--437, (1992).
 */

#include "gg.h"
#include "include/define.h"

static int preColussi(unsigned char * x, int m, int h[], int next[], int shift[]) 
{
    int i, k, nd, q, r, s;
    int hmax[XSIZE], kmin[XSIZE], nhd0[XSIZE], rmin[XSIZE];

    /* Computation of hmax */
    i = k = 1;
    do {
        while (x[i] == x[i - k])
            i++;
        hmax[k] = i;
        q = k + 1;
        while (hmax[q - k] + k < i) {
            hmax[q] = hmax[q - k] + k;
            q++;
        }
        k = q;
        if (k == i + 1)
            i = k;
    } while (k <= m);

    /* Computation of kmin */
    memset(kmin, 0, m * sizeof(int));
    for (i = m; i >= 1; --i)
        if (hmax[i] < m)
            kmin[hmax[i]] = i;

    /* Computation of rmin */
    for (i = m - 1; i >= 0; --i) {
        if (hmax[i + 1] == m)
            r = i + 1;
        if (kmin[i] == 0)
            rmin[i] = r;
        else
            rmin[i] = 0;
    }

    /* Computation of h */
    s = -1;
    r = m;
    for (i = 0; i < m; ++i)
        if (kmin[i] == 0)
            h[--r] = i;
        else
            h[++s] = i;
    nd = s;

    /* Computation of shift */
    for (i = 0; i <= nd; ++i)
        shift[i] = kmin[h[i]];
    for (i = nd + 1; i < m; ++i)
        shift[i] = rmin[h[i]];
    shift[m] = rmin[0];

    /* Computation of nhd0 */
    s = 0;
    for (i = 0; i < m; ++i) {
        nhd0[i] = s;
        if (kmin[i] > 0)
            ++s;
    }

    /* Computation of next */
    for (i = 0; i <= nd; ++i)
        next[i] = nhd0[h[i] - kmin[h[i]]];
    for (i = nd + 1; i < m; ++i)
        next[i] = nhd0[m - rmin[h[i]]];
    next[m] = nhd0[m - rmin[h[m - 1]]];

    return (nd);
}

void gg(search_parameters params) {
    int i, j, k, ell, last, nd;
    int h[XSIZE], next[XSIZE], shift[XSIZE];
    char heavy;

    for (ell = 0; 
        params.pattern[ell] == params.pattern[ell + 1] && 
        ell < params.pattern_size; ell++);
    if (ell == params.pattern_size - 1)
        /* Searching for a power of a single character */
        for (j = ell = 0; j < params.text_size; ++j)
            if (params.pattern[0] == params.text[j]) {
                ++ell;
                if (ell >= params.pattern_size)
                    params.match[j - params.pattern_size + 1] = 1;
            }
            else
                ell = 0;
    else {
        /* Preprocessing */
        nd = preColussi(params.pattern, params.pattern_size, h, next, shift);

        /* Searching */
        i = j = heavy = 0;
        last = -1;
        while (j <= params.text_size - params.pattern_size) {
            if (heavy && i == 0) {
                k = last - j + 1;
                while (params.pattern[0] == params.text[j + k]) ++k;
                if (k <= ell || params.pattern[ell + 1] != params.text[j + k]) {
                    i = 0;
                    j += (k + 1);
                    last = j - 1;
                } else {
                    i = 1;
                    last = j + k;
                    j = last - (ell + 1);
                }
                heavy = 0;
            } else {
                while (i < params.pattern_size && 
                    last < j + h[i] && 
                    params.pattern[h[i]] == params.text[j + h[i]]) ++i;
                if (i >= params.pattern_size || last >= j + h[i]) {
                    params.match[j] = 1;
                    i = params.pattern_size;
                }
                if (i > nd)
                    last = j + params.pattern_size - 1;
                j += shift[i];
                i = next[i];
            }
            heavy = (j > last ? 0 : 1);
        }
    }
}
