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
 * This is an implementation of the Average Optimal Shift Or algorithm
 * in K. Fredriksson and S. Grabowski.
 * Practical and Optimal String Matching. SPIRE, Lecture Notes in Computer Science, vol.3772, pp.376--387, Springer-Verlag, Berlin, (2005).
 */

#include "aoso2.h"
#include "include/define.h"
#include "include/log2.h"

static void verify(search_parameters params, int j, int q, unsigned int D, unsigned int mm) {
    unsigned int s;
    int c, k, i;
    D = (D & mm)^mm;
    while (D != 0) {
        s = LOG2(D); 
        c = -(params.pattern_size/q-1)*q-s/(params.pattern_size/q);
        k = 0; 
        i= j + c;
        if(i>=0 && i<=params.text_size-params.pattern_size) 
            while(k<params.pattern_size && params.pattern[k]==params.text[i+k]) k++;
        if (k==params.pattern_size) params.match[i] = 1;
        D &= ~(1<<s);
    }
}



static void verify_large(search_parameters params, int j, int q, unsigned int D, unsigned int mm, int p_len) {
    unsigned int s;
    int c, k, i;
    D = (D & mm)^mm;
    while (D != 0) {
        s = LOG2(D);
        c = -(p_len/q-1)*q-s/(p_len/q);
        k = 0; 
        i= j + c;
        if(i>=0 && i<=params.text_size-params.pattern_size) while(k<params.pattern_size && params.pattern[k]==params.text[i+k]) k++;
        if(k==params.pattern_size) params.match[i] = 1;
        D &= ~(1<<s);
    }
}

static void aoso2_large(search_parameters params) {
    unsigned int B[SIGMA], D, h, mm;
    int i, j, p_len;
    int q = 2;
    p_len = 32;

    /* Preprocessing */
    for (i = 0; i < SIGMA; ++i) B[i] = ~0; 
    h = mm = 0;
    for (j = 0; j < q; ++j) {
        for (i = 0; i < p_len/q; ++i) {
            B[params.pattern[i*q+j]] &= ~(1<<h);
            ++h;
        }
        mm |= (1<<(h-1));
    }

    /* Searching */
    D = ~0;
    j = 0;
    while (j < params.text_size) {
        D = ((D & ~mm)<<1)|B[params.text[j]];
        if ((D & mm) != mm)
            verify_large(params, j, q, D, mm, p_len);
        j += q;
    }
}

void aoso2(search_parameters params) 
{
    unsigned int B[SIGMA], D, h, mm;
    int i, j;
    int q=2;

    if (params.pattern_size<=q) return;
    if (params.pattern_size>32) {
        aoso2_large(params);
        return;
    }

    /* Preprocessing */
    for (i = 0; i < SIGMA; ++i) B[i] = ~0;
    h = mm = 0;
    for (j = 0; j < q; ++j) {
        for (i = 0; i < params.pattern_size/q; ++i) {
            B[params.pattern[i*q+j]] &= ~(1<<h);
            ++h;
        }
        mm |= (1<<(h-1));
    }

    /* Searching */
    D = ~0;
    j = 0;
    while (j < params.text_size) {
        D = ((D & ~mm)<<1)|B[params.text[j]];
        if ((D & mm) != mm)
            verify(params, j, q, D, mm);
        j += q;
    }
}
