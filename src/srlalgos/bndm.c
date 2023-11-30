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
 * This is an implementation of the Backward Nondeterministic DAWG Matching algorithm
 * in G. Navarro and M. Raffinot.
 * A Bit-parallel Approach to Suffix Automata: Fast Extended String Matching. n.TR/DC--98--1, (1998).
 */

#include "bndm.h"

#include <string.h>
#include <stdlib.h>

void bndm(search_parameters params)
{
    int B[SIGMA];
    int i, j, s, D, last, count;

    if (params.pattern_size > 32){
        bndm_large(params);
        return;
    }

    /* Preprocessing */
    for (i=0; i<SIGMA; i++) B[i]=0;
    s=1;
    for (i=params.pattern_size-1; i>=0; i--) {
        B[params.pattern[i]] |= s;
        s <<= 1;
    }

    /* Searching */
    j=0;
    count=0;
    while (j <= params.text_size-params.pattern_size) {
        i=params.pattern_size-1;
        last=params.pattern_size;
        D = ~0;
        while (i>=0 && D!=0) {
            D &= B[params.text[j+i]];
            i--;
            if (D != 0) {
                if (i >= 0) last = i+1;
                else params.match[j] = 1;
            }
            D <<= 1;
        }
        j += last;
    }
}

/*
 * Backward Nondeterministic DAWG Matching designed for large patterns
 * The present implementation searches for prefixes of the pattern of length 32.
 * When an occurrence is found the algorithm tests for the whole occurrence of the pattern
 */

void bndm_large(search_parameters params)
{
    int B[SIGMA];
    int i, j, s, D, last, count, p_len, k;

    p_len = params.pattern_size;
    params.pattern_size = 32;

    /* Preprocessing */
    memset(B,0,SIGMA*sizeof (int) );
    s=1;
    for (i=params.pattern_size-1; i>=0; i--) {
        B[params.pattern[i]] |= s;
        s <<= 1;
    }

    /* Searching */
    j=0;
    count=0;
    while (j <= params.text_size-params.pattern_size) {
        i=params.pattern_size-1;
        last=params.pattern_size;
        D = ~0;
        while (i>=0 && D!=0) {
            D &= B[params.text[j+i]];
            i--;
            if (D != 0) {
                if (i >= 0)
                    last = i+1;
                else {
                    k = params.pattern_size;
                    while (k<p_len && params.pattern[k]==params.text[j+k]) k++;
                    if (k==p_len) params.match[j] = 1;
                }
            }
            D <<= 1;
        }
        j += last;
    }
}
