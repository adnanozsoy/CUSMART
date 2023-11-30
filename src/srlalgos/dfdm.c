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
 * This is an implementation of the Double Forward DAWG Matching algorithm
 * in C. Allauzen and M. Raffinot.
 * Simple Optimal String Matching Algorithm. J. Algorithms, vol.36, pp.102-116, (2000).
 */

#include "dfdm.h"
#include "include/AUTOMATON.h"

void dfdm(search_parameters params)
{
    int init, state1, state2, ell, count;
    int alpha, beta;
    int *ttrans, *tlength, *tsuffix;
    unsigned char *tterminal;
    int i, j, s, end, nMinusM, logM, temp;

    /* Preprocessing */
    ttrans = (int *)malloc(3*params.pattern_size*SIGMA*sizeof(int));
    memset(ttrans, -1, 3*params.pattern_size*SIGMA*sizeof(int));
    tlength = (int *)calloc(3*params.pattern_size, sizeof(int));
    tsuffix = (int *)calloc(3*params.pattern_size, sizeof(int));
    tterminal = (char *)calloc(3*params.pattern_size, sizeof(char));
    buildSimpleSuffixAutomaton(params.pattern, params.pattern_size, ttrans, tlength, tsuffix, tterminal);
    init = 0;

    /* Searching */
    count = 0;
    logM = 0;
    temp = params.pattern_size;
    int a = 2;
    while (temp > a) {
        ++logM;
        temp /= a;
    }
    ++logM;

    beta = params.pattern_size-1-MAX(1,MIN(params.pattern_size/5, 3*logM));
    alpha = MIN(params.pattern_size/2,beta-1);
    s = 0;
    ell = 0;
    state2 = init;
    nMinusM = params.text_size-params.pattern_size;
    while (s <= nMinusM) {
        state1 = init;
        j = s+beta;
        end = s+params.pattern_size;
        while ((j < end) && (getTarget(state1, params.text[j]) != UNDEFINED)) {
            state1 = getTarget(state1, params.text[j]);
            ++j;
        }

        if (j < s+params.pattern_size) {
            state2 = getSuffixLink(state1);
            while ((state2 != init) && (getTarget(state2, params.text[j]) == UNDEFINED)) {
                state2 = getSuffixLink(state2);
            }
            if (getTarget(state2, params.text[j]) != UNDEFINED) {
                ell = getLength(state2) + 1;
                state2 = getTarget(state2, params.text[j]);
            }
            else {
                ell = 0;
                state2 = init;
            }
            ++j;
        }
        else {
            j = s+ell;
        }
        end = s+params.pattern_size;
        while (((j < end) || (ell < alpha)) && (j < params.text_size)) {
            if (getTarget(state2, params.text[j]) != UNDEFINED) {
                ++ell;
                state2 = getTarget(state2, params.text[j]);
            }
            else {
                while ((state2 != init) && (getTarget(state2, params.text[j]) == UNDEFINED)) {
                    state2 = getSuffixLink(state2);
                }
                if (getTarget(state2, params.text[j]) != UNDEFINED) {
                    ell = getLength(state2) + 1;
                    state2 = getTarget(state2, params.text[j]);
                }
                else {
                    ell = 0;
                    state2 = init;
                }
            }
            if (ell == params.pattern_size) {
                params.match[j-params.pattern_size+1] = 1;
                state2 = getSuffixLink(state2);
                ell = getLength(state2);
            }
            ++j;
        }
        s = j-ell;
    }
    free(ttrans);
    free(tlength);
    free(tsuffix);
    free(tterminal);
}
