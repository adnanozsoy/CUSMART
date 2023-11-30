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
 * This is an implementation of the Brute Force algorithm
 * in T. H. Cormen and C. E. Leiserson and R. L. Rivest and C. Stein. 
 * Introduction to Algorithms. MIT Press, (2001).
 */

#include "bf.h"
#include "include/define.h"


void bf(search_parameters params) {
    int i, j;
    /* Searching */
    for (j = 0; j <= params.text_size-params.pattern_size; ++j) {
        for (i = 0; i < params.pattern_size && 
        	params.pattern[i] == params.text[i + j]; ++i);
        if (i >= params.pattern_size) 
        	params.match[j] = 1;
    }
}

