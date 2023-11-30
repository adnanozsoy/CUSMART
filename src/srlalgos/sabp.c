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
 * This is an implementation of the Small Alphabet Bit Parallel algorithm
 * in G. Zhang and E. Zhu and L. Mao and M. Yin. 
 * A Bit-Parallel Exact String Matching Algorithm for Small Alphabet. 
 * Proceedings of the Third International Workshop on Frontiers in Algorithmics, FAW 2009, Hefei, China, 
 * Lecture Notes in Computer Science, vol.5598, pp.336--345, Springer-Verlag, Berlin, (2009).
 */

#include "sabp.h"
#include "include/define.h"

int pow2(int n) {
   int p, i;
   p = 1;
   i = 0;
   while (i < n) { 
      p *= 2;      
      ++i;         
   } 
   return p;
}

int mylog2(int unsigned n) {
   int ell;
   ell = 0;
   while (n >= 2) {
      ++ell;
      n /= 2;
   }
   return ell;
}

void sabp_large(search_parameters params) {
  int i, j, z, k, first, p_len, m;
  unsigned int b, D, Delta, mask, mask2, T[SIGMA];
  p_len = params.pattern_size;
  m = 30;
  
  /* Preprocessing */
  z = WORD;
  mask = 1;
  for (i = 1; i < m; ++i) mask = (mask << 1) | 1;
  
  for (i = 0; i < SIGMA; ++i) T[i] = mask;
  mask2 = 1;
  for (i = 0; i < m; ++i) {
    T[params.pattern[i]] &= ~mask2;
    mask2 <<= 1;
  }
  mask2 >>= 1;
  
  /* searching */
  D = 0;
  b = mask;
  i = m - 1;
  j = i;
  while (i < params.text_size) {
    D |= (T[params.text[j]] << (i - j));
    D &= mask;
    b &= ~pow2(m - i + j - 1);
    if ((D & mask2) == 0) {
      if (b == 0) {
	D |= mask2;
	k=0;
	first = i-m+1;
	while(k<p_len && params.pattern[k]==params.text[first+k]) k++;
	if (k==p_len) params.match[i] = 1;
      }
      else {
	j = i - (m - mylog2(b) - 1);
	continue;
      }
    }
    if (D == mask) {
      D = 0;
      b = mask;
      i += m;
    }
    else {
      Delta = m - mylog2(~D&mask) - 1;
      D <<= Delta;
      b = ((b | ~mask) >> Delta) & mask;
      i += Delta;
    }
    j = i;
  }
}


void sabp(search_parameters params) {
  int i, j, z;
  unsigned int b, D, Delta, mask, mask2, T[SIGMA];
  if (params.pattern_size>30) {
    sabp_large(params);   
    return;
  }
  
  /* Preprocessing */
  z = WORD;
  mask = 1;
  for (i = 1; i < params.pattern_size; ++i) mask = (mask << 1) | 1;
  for (i = 0; i < SIGMA; ++i) T[i] = mask;
  mask2 = 1;
  for (i = 0; i < params.pattern_size; ++i) {
    T[params.pattern[i]] &= ~mask2;
    mask2 <<= 1;
  }
  mask2 >>= 1;
  
  /* Searching */
  D = 0;
  b = mask;
  i = params.pattern_size - 1;
  j = i;
  while (i < params.text_size) {
    D |= (T[params.text[j]] << (i - j));
    D &= mask;
    b &= ~pow2(params.pattern_size - i + j - 1);
    if ((D & mask2) == 0) {
      if (b == 0) {
	D |= mask2;
	params.match[i - params.pattern_size + 1] = 1;
      }
      else {
	j = i - (params.pattern_size - mylog2(b) - 1);
	continue;
      }
    }
    if (D == mask) {
      D = 0;
      b = mask;
      i += params.pattern_size;
    }
    else {
      Delta = params.pattern_size - mylog2(~D & mask) - 1;
      D <<= Delta;
      b = ((b | ~mask) >> Delta) & mask;
      i += Delta;
    }
    j = i;
  }
}

