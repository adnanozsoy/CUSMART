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
 * This is an implementation of the Improved Linear DAWG Matching 2 algorithm
 * in C. Liu and Y. Wang and D. Liu and D. Li.
 * Two Improved Single Pattern Matching Algorithms. ICAT Workshops, pp.419--422, IEEE Computer Society, Hangzhou, China, (2006).
 */

#include "ildm2.h"
#include "include/define.h"
#include "include/AUTOMATON.h"

#include <stdlib.h>
#include <string.h>


void ildm2(search_parameters params)
{
    int *ttrans, *tlength, *tsuffix;
    int *ttransSMA;
    unsigned char *tterminal;
    unsigned char *xR;

    xR = (char*) malloc (sizeof(char)*(params.pattern_size+1));
    for (int i=0; i<params.pattern_size;
         i++) xR[i] = params.pattern[params.pattern_size-i-1];
    xR[params.pattern_size] = '\0';

    /* Preprocessing */
    ttrans = (int *)malloc(3*params.pattern_size*SIGMA*sizeof(int));
    memset(ttrans, -1, 3*params.pattern_size*SIGMA*sizeof(int));
    tlength = (int *)calloc(3*params.pattern_size, sizeof(int));
    tsuffix = (int *)calloc(3*params.pattern_size, sizeof(int));
    tterminal = (char *)calloc(3*params.pattern_size, sizeof(char));
    buildSimpleSuffixAutomaton(xR, params.pattern_size, ttrans, tlength, tsuffix,
                               tterminal);

    ttransSMA = (int *)malloc((params.pattern_size+1)*SIGMA*sizeof(int));
    memset(ttransSMA, -1, (params.pattern_size+1)*SIGMA*sizeof(int));
    preSMA(params.pattern, params.pattern_size, ttransSMA);

    /* Searching */
    int k = params.pattern_size-1;
    while ( k < params.text_size ) {
        int L = 0 ;
        int R = 0;
        int l = 0;
        while ( k-l >= 0 && ( L = getTarget(L, params.text[k-l]) ) != UNDEFINED ) {
            l++;
            if ( isTerminal(L) ) R = l ;
        }
        while ( R > params.pattern_size/2 ) {
            if ( R==params.pattern_size )
                params.match[k-params.pattern_size+1] = 1;
            k++ ;
            if ( k >= params.text_size ) break;
            R = getSMA(R, params.text[k]) ;
        }
        k = k + params.pattern_size - R ;
    }

    free(ttransSMA);
    free(ttrans);
    free(tlength);
    free(tsuffix);
    free(tterminal);
}
