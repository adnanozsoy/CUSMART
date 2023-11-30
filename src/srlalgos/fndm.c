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
 * This is an implementation of the Forward Nondeterministic DAWG Matching algorithm
 * in J. Holub and B. Durian.
 * Talk: Fast variants of bit parallel approach to suffix automata. 
 * The Second Haifa Annual International Stringology Research Workshop of the Israeli Science Foundation, (2005).
 */

#include "fndm.h"
#include "include/define.h"

static void fndm_large(search_parameters);

void fndm(search_parameters params)
{
    unsigned int D, B[SIGMA], NEG0, NEG0m1;
    int i, j, k, first;
    if (params.pattern_size>32) {
        fndm_large(params);
        return;
    }

    /* Preprocessing */
    for (i=0; i<SIGMA; i++) B[i] = ~0;
    for (j=0; j<params.pattern_size; j++) B[params.pattern[j]] = B[params.pattern[j]] & ~(1<<j);
    NEG0 = ~0;
    NEG0m1 = ~0<<(params.pattern_size-1);

    /* Searching */
    i = params.pattern_size-1;
    while ( i < params.text_size ) {
        D = B[params.text[i]];
        while (D != NEG0) {
            if (D < NEG0m1) {
                k = 0;
                first=i-params.pattern_size+1;
                while (k<params.pattern_size && params.pattern[k]==params.text[first+k]) k++;
                if (k==params.pattern_size && i<params.text_size) params.match[first] = 1;
            }
            i = i+1;
            D = (D<<1) | B[params.text[i]];
        }
        i=i+params.pattern_size;
    }
}

/*
 * Forward Nondeterministic DAWG Matching algorithm designed for large patterns
 * The present implementation searches for prefixes of the pattern of length 32.
 * When an occurrence is found the algorithm tests for the whole occurrence of the pattern
 */

static void fndm_large(search_parameters params)
{
    unsigned int D, B[SIGMA], NEG0, NEG0m1;
    int i, j, k, first, p_len;

    p_len = params.pattern_size;
    params.pattern_size = 32;

    /* Preprocessing */
    for (i=0; i<SIGMA; i++) B[i] = ~0;
    for (j=0; j<params.pattern_size; j++) B[params.pattern[j]] = B[params.pattern[j]] & ~(1<<j);
    NEG0 = ~0;
    NEG0m1 = ~0<<(params.pattern_size-1);

    /* searching */
    i = params.pattern_size-1;
    while ( i < params.text_size ) {
        D = B[params.text[i]];
        while (D != NEG0) {
            if (D < NEG0m1) {
                k = 0;
                first=i-params.pattern_size+1;
                while (k<p_len && params.pattern[k]==params.text[first+k]) k++;
                if (k==p_len && i<params.text_size) params.match[first] = 1;
            }
            i = i+1;
            D = (D<<1) | B[params.text[i]];
        }
        i=i+params.pattern_size;
    }
}
