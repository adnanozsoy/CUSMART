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
 * This is an implementation of the Bitparallel Wide Window algorithm
 * in L. He and B. Fang and J. Sui.
 * The wide window string matching algorithm. Theor. Comput. Sci., vol.332, n.1-3, pp.391--404, Elsevier Science Publishers Ltd., Essex, UK, (2005).
 */

#include "include/define.h"
#include "bww.h"
#include <stdlib.h>
#include <string.h>

static void bww_large(search_parameters params);

void bww(search_parameters params)
{
    int i, j, k, left, r, ell, end, count;
    unsigned int B[SIGMA], C[SIGMA], s, t, R, L;
    unsigned int pre, cur;

    if (params.pattern_size>30) {
        bww_large(params);
        return;
    }

    /* Preprocessing */
    /* Left to right automaton */
    memset(B, 0, SIGMA*sizeof(int));
    s = 1;
    for (i = 0; i < params.pattern_size; ++i) {
        B[params.pattern[i]] |= s;
        s <<= 1;
    }
    s >>= 1;

    /* Right to left automaton */
    memset(C, 0, SIGMA*sizeof(int));
    t = 1;
    for (i = params.pattern_size-1; i >= 0; --i) {
        C[params.pattern[i]] |= t;
        t <<= 1;
    }
    t >>= 1;

    /* Searching */
    count = 0;
    end = params.text_size/params.pattern_size;
    if (params.text_size%params.pattern_size > 0) ++end;
    for (k = 1; k < end; ++k) {
        /* Left to right scanning */
        r = pre = left = 0;
        R = ~0;
        cur = s;
        while (R != 0 && k*params.pattern_size-1+r < params.text_size) {
            R &= B[params.text[k*params.pattern_size-1+r]];
            ++r;
            if ((R & s) != 0) {
                pre |= cur;
                left = MAX(left, params.pattern_size+1-r);
            }
            R <<= 1;
            cur >>= 1;
        }
        /* Right to left scanning */
        L = ~0;
        cur = 1;
        ell = 0;
        while (L != 0 && left > ell) {
            L &= C[params.text[k*params.pattern_size-1-ell]];
            if ((L&t) != 0 && (cur&pre) != 0) params.match[k*params.pattern_size-1-ell] = 1;
            L <<= 1;
            cur <<= 1;
            ++ell;
        }
    }
    /* Test the last portion of the text */
    for (j = (end-1)*params.pattern_size;
         j <= params.text_size - params.pattern_size; ++j) {
        for (i = 0; i < params.pattern_size
             && params.pattern[i] == params.text[i + j]; ++i);
        if (i >= params.pattern_size) params.match[i] = 1;
    }
}

/*
 * Bitparallel Wide Window algorithm designed for large patterns
 * The present implementation searches for prefixes of the pattern of length 32.
 * When an occurrence is found the algorithm tests for the whole occurrence of the pattern
 */

static void bww_large(search_parameters params)
{
    int i, k, left, r, ell, end, count, p_len, first, j;
    unsigned int B[SIGMA], C[SIGMA], s, t, R, L;
    unsigned int pre, cur;

    p_len=params.pattern_size;
    params.pattern_size=30;

    /* Preprocessing */
    /* Left to right automaton */
    memset(B, 0, SIGMA*sizeof(int));
    s = 1;
    for (i = 0; i < params.pattern_size; ++i) {
        B[params.pattern[i]] |= s;
        s <<= 1;
    }
    s >>= 1;

    /* Right to left automaton */
    memset(C, 0, SIGMA*sizeof(int));
    t = 1;
    for (i = params.pattern_size-1; i >= 0; --i) {
        C[params.pattern[i]] |= t;
        t <<= 1;
    }
    t >>= 1;

    /* Searching */
    count = 0;
    end = params.text_size/params.pattern_size;
    if (params.text_size%params.pattern_size > 0) ++end;
    for (k = 1; k < end; ++k) {
        /* Left to right scanning */
        r = pre = left = 0;
        R = ~0;
        cur = s;
        while (R != 0 && k*params.pattern_size-1+r < params.text_size) {
            R &= B[params.text[k*params.pattern_size-1+r]];
            ++r;
            if ((R & s) != 0) {
                pre |= cur;
                left = MAX(left, params.pattern_size+1-r);
            }
            R <<= 1;
            cur >>= 1;
        }
        /* Right to left scanning */
        L = ~0;
        cur = 1;
        ell = 0;
        while (L != 0 && left > ell) {
            L &= C[params.text[k*params.pattern_size-1-ell]];
            if ((L&t) != 0 && (cur&pre) != 0) {
                j = params.pattern_size;
                first = k*params.pattern_size-1-ell;
                while (j<p_len && params.pattern[j]==params.text[first+j]) j++;
                if (j==p_len) params.match[first] = 1;
            }
            L <<= 1;
            cur <<= 1;
            ++ell;
        }
    }
    for (j = (end-1)*params.pattern_size;
         j <= params.text_size - params.pattern_size; ++j) {
        for (i = 0; i < params.pattern_size
             && params.pattern[i] == params.text[i + j]; ++i);
        if (i >= params.pattern_size) params.match[i] = 1;
    }
}




