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
 * This is an implementation of the Horspool algorithm
 * in R. N. Horspool.
 * Practical fast searching in strings. Softw. Pract. Exp., vol.10, n.6, pp.501--506, (1980).
 */

#include "hor.h"
#include "include/define.h"

#include "stdio.h"
#include "stdlib.h"
#include "string.h"

static void pre_horspool(char *pattern, int pattern_size, unsigned char hbc[]) {
  int i;
  for (i=0;i<SIGMA;i++)   hbc[i]=pattern_size;
  for (i=0;i<pattern_size-1;i++) hbc[pattern[i]]=pattern_size - i - 1;
}

void hor(search_parameters params) {
  int i, s;
  unsigned char hbc[SIGMA];
  
  pre_horspool(params.pattern, params.pattern_size, hbc);

  /* Searching */
  s = 0;
  while(s <= params.text_size - params.pattern_size) {
    i=0;
    while(i < params.pattern_size && params.pattern[i] == params.text[s + i]) i++;
    if (i == params.pattern_size) params.match[0]++;
    s += hbc[params.text[s + params.pattern_size - 1]];
  }
}
