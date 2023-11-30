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
 * This is an implementation of the Forward Semplified BNDM algorithm using q-grams
 * in Hannu Peltola and Jorma Tarhio
 * Variations of Forward-SBNDM
 * Proceedings of the Prague Stringology Conference 2011, pp.3--14, Czech Technical University in Prague, Czech Republic, (2008).
 * Q is the dimension of q-grams
 * F is the number of forward characters
 */

#include "fsbndmq20.h"
#include "include/define.h"

#include <stdlib.h>
#include <string.h>

#define Q 2
#define F 0

void fsbndmq20(search_parameters params)
{
    unsigned int B[SIGMA], D, set;
    int i, j, pos, mm, sh, m1, count;

    if (params.pattern_size<Q) return -1;
    int plen = params.pattern_size;
    int larger = params.pattern_size+F>WORD? 1:0;
    if (larger) params.pattern_size = WORD-F;

    /* Preprocessing */
    count = 0;
    set = 0;
    for (j=0; j<F; j++) set = (set << 1) | 1;
    for (i=0; i<SIGMA; i++) B[i]=set;
    for (i = 0; i < params.pattern_size;
         ++i) B[params.pattern[i]] |= (1<<(params.pattern_size-i-1+F));
    mm = params.pattern_size-Q+F;
    sh = params.pattern_size-Q+F+1;
    m1 = params.pattern_size-1;

    /* Searching */
    // if (!memcmp(params.pattern, params.text, params.pattern_size)) params.match[0] = 1;
    int end = params.text_size-plen+params.pattern_size;
    j = params.pattern_size;
    while (j < end) {
        D = B[params.text[j]];
        D = (D<<1) & B[params.text[j-1]];
        if (D != 0) {
            pos = j;
            while (D = (D<<1) & B[params.text[j-2]]) --j;
            j += mm;
            if (j == pos) {
                if (larger) {
                    i=params.pattern_size;
                    while (i<plen && params.pattern[i]==params.text[j-m1+i]) i++;
                    if (i==plen) params.match[j] = 1;
                }
                else params.match[j] = 1;
                ++j;
            }
        }
        else j += sh;
    }
}