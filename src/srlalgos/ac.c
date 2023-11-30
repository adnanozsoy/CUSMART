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
 * This is an implementation of the Apostolico-Crochemore algorithm
 * in A. Apostolico and M. Crochemore. 
 * Optimal canonization of all substrings of a string. Inf. Comput., vol.95, n.1, pp.76--95, (1991).
 */

#include "include/define.h"
#include "ac.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"

static void preKmp(unsigned char *pattern, int pattern_size, int kmpNext[])
{
    int i, j;
    i = 0;
    j = kmpNext[0] = -1;
    while (i < pattern_size) {
        while (j > -1 && pattern[i] != pattern[j])
            j = kmpNext[j];
        i++;
        j++;
        if (i < pattern_size && pattern[i] == pattern[j])
            kmpNext[i] = kmpNext[j];
        else
            kmpNext[i] = j;
    }
}
void ac(search_parameters params){

    int i, j, k, ell, kmpNext[XSIZE];

    /* Preprocessing */
    preKmp (params.pattern, params.pattern_size, kmpNext);
    for (ell = 1; ell < params.pattern_size && params.pattern[ell - 1] == params.pattern[ell]; ell++);
    if (ell == params.pattern_size)
        ell = 0;

    /* Searching */
    i = ell;
    j = k = 0;
    while (j <= params.text_size - params.pattern_size) {
        while (i < params.pattern_size && params.pattern[i] == params.text[i + j])
            ++i;
        if (i >= params.pattern_size) {
            while (k < ell && params.pattern[k] == params.text[j + k])
                ++k;
            if (k >= ell)
                params.match[j] = 1;
        }
        j += (i - kmpNext[i]);
        if (i == ell)
            k = MAX (0, k - 1);

        else if (kmpNext[i] <= ell) {
            k = MAX (0, kmpNext[i]);
            i = ell;
        }
        else {
            k = ell;
            i = kmpNext[i];
        }
    }
}


