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
 * This is an implementation of the Skip Search algorithm
 * in C. Charras and T. Lecroq and J. D. Pehoushek.
 * A Very Fast String Matching Algorithm for Small Alphabets and Long Patterns. 
 * Proceedings of the 9th Annual Symposium on Combinatorial Pattern Matching, Lecture Notes in Computer Science, n.1448, pp.55--64, Springer-Verlag, Berlin, rutgers, (1998).
 */

#include "skip.h"
#include "include/define.h"
#include "include/AUTOMATON.h"

#include <string.h>
#include <stdlib.h>

void skip(search_parameters params) {
    int i, j, h, k;
    List ptr, z[SIGMA];
    
    /* Preprocessing */
    memset(z, 0, SIGMA*sizeof(List));
    for (i = 0; i < params.pattern_size; ++i) {
        ptr = (List)malloc(sizeof(struct _cell));
        ptr->element = i;
        ptr->next = z[params.pattern[i]];
        z[params.pattern[i]] = ptr;
    }

    /* Searching */
    for (j = params.pattern_size - 1; j < params.text_size; j += params.pattern_size){
        for (ptr = z[params.text[j]]; ptr != NULL; ptr = ptr->next){ 
            if ((j-ptr->element) <= params.text_size - params.pattern_size) {
                k = 0;
                h = j-ptr->element;
                while(k<params.pattern_size && params.pattern[k]==params.text[h+k]) k++;
                    if (k>=params.pattern_size) params.match[j] = 1;
            }
        }
    }
}
