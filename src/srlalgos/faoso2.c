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
 * This is an implementation of the Fast Average Optimal Shift Or algorithm
 * in K. Fredriksson and S. Grabowski.
 * Practical and Optimal String Matching. SPIRE, Lecture Notes in Computer Science, vol.3772, pp.376--387, Springer-Verlag, Berlin, (2005).
 */

#include "faoso2.h"
#include "include/define.h"
#include "include/log2.h"


static void verify(search_parameters params, int j, int q, int u,
                   unsigned int D, unsigned int mm)
{
    int s, c, mq, v, z, i, k;

    D = (D & mm)^mm;
    mq = params.pattern_size/q-1;
    while (D != 0) {
        s = LOG2(D);
        v = mq+u;
        c = -mq*q;
        z = s%v-mq;
        c -= (s/v + z*q);
        i = j+c;
        k = 0;
        if (i>=0 && i<=params.text_size-params.pattern_size)
            while (k<params.pattern_size && params.pattern[k]==params.text[i+k]) k++;
        if (k==params.pattern_size) params.match[i] = 1;
        D &= ~(1<<s);
    }
}

/*
 * Fast Average Optimal Shift Or algorithm designed for large patterns
 * The present implementation searches for prefixes of the pattern of length 32.
 * When an occurrence is found the algorithm tests for the whole occurrence of the pattern
 */

static void verify_large(search_parameters params, int j, int q, int u,
                         unsigned int D, unsigned int mm, int p_len)
{
    int s, c, mq, v, z, i, k;

    D = (D & mm)^mm;
    mq = params.pattern_size/q-1;
    while (D != 0) {
        s = LOG2(D);
        v = mq+u;
        c = -mq*q;
        z = s%v-mq;
        c -= (s/v + z*q);
        i = j+c;
        k = 0;
        if (i>=0 && i<=params.text_size-params.pattern_size)
            while (k<p_len && params.pattern[k]==params.text[i+k]) k++;
        if (k==params.pattern_size) params.match[i] = 1;
        D &= ~(1<<s);
    }
}

static void search_large(search_parameters params)
{
    unsigned int B[SIGMA], D, h, mm;
    unsigned int masq;
    int i, j, u, count, p_len;
    int uq, uq1, mq;
    int q = 2;

    u = 2;
    p_len = params.pattern_size;
    params.pattern_size = 32-u+1;

    /* Preprocessing */
    masq = 0;
    mq = params.pattern_size/q;
    h = mq;
    for (j = 0; j < q; ++j) {
        masq |= (1<<h);
        masq |= (1<<h);
        h += mq;
        ++h;
    }
    for (i = 0; i < SIGMA; ++i)
        B[i] = ~0;
    h=mm=0;
    for (j = 0; j < q; ++j) {
        for (i = 0; i < mq; ++i) {
            B[params.pattern[i*q+j]] &= ~(1<<h);
            ++h;
        }
        mm |= (1<<(h-1));
        ++h;
        mm |= (1<<(h-1));
        ++h;
        --h;
    }

    /* Searching */
    count = 0;
    D = ~mm;
    j = 0;
    uq = u*q;
    uq1 = (u-1)*q;
    while (j < params.text_size) {
        D = (D<<1)|(B[params.text[j]]&~masq);
        D = (D<<1)|(B[params.text[j+q]]&~masq);
        if ((D & mm) != mm)
            verify_large(params, j+uq1, q, u, D, mm, p_len);
        D &= ~mm;
        j += uq;
    }
}

void faoso2(search_parameters params)
{
    unsigned int B[SIGMA], D, h, mm;
    unsigned int masq;
    int i, j, u, count;
    int uq, uq1, mq;
    int q=2;

    u = 2;
    if (params.pattern_size>32-u+1) {
        search_large(params);
        return;
    }
    if (params.pattern_size<=q) return;

    masq = 0;
    mq = params.pattern_size/q;
    h = mq;
    for (j = 0; j < q; ++j) {
        masq |= (1<<h);
        masq |= (1<<h);
        h += mq;
        ++h;
    }
    for (i = 0; i < SIGMA; ++i)
        B[i] = ~0;
    h=mm=0;
    for (j = 0; j < q; ++j) {
        for (i = 0; i < mq; ++i) {
            B[params.pattern[i*q+j]] &= ~(1<<h);
            ++h;
        }
        mm |= (1<<(h-1));
        ++h;
        mm |= (1<<(h-1));
        ++h;
        --h;
    }

    /* Searching */
    count = 0;
    D = ~mm;
    j = 0;
    uq = u*q;
    uq1 = (u-1)*q;
    while (j < params.text_size) {
        D = (D<<1)|(B[params.text[j]]&~masq);
        D = (D<<1)|(B[params.text[j+q]]&~masq);
        if ((D & mm) != mm)
            verify(params, j+uq1, q, u, D, mm);
        D &= ~mm;
        j += uq;
    }
}
