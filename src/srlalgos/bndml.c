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
 * This is an implementation of the BNDM for Long patterns
 * in G. Navarro and M. Raffinot. 
 * Fast and Flexible String Matching by Combining Bit-Parallelism and Suffix Automata. ACM J. Experimental Algorithmics, vol.5, pp.4, (2000).
 */

#include "include/define.h"
#include "bndml.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHAR_BIT 8
#define WORD_TYPE unsigned int
#define WORD_BITS (sizeof(WORD_TYPE)*CHAR_BIT)
#define bit_size(bits) (((bits)+WORD_BITS-1)/WORD_BITS)
#define bit_byte(bit) ((bit) / WORD_BITS)
#define bit_mask(bit) (1L << ((bit) % WORD_BITS))
#define bit_alloc(bits) calloc(bit_size(bits), sizeof(WORD_TYPE))
#define bit_set(name, bit) ((name)[bit_byte(bit)] |= bit_mask(bit))
#define bit_clear(name, bit) ((name)[bit_byte(bit)] &= ~bit_mask(bit))
#define bit_test(name, bit) ((name)[bit_byte(bit)] & bit_mask(bit))
#define bit_zero(name, bits) memset(name, 0, bit_size(bits) * sizeof(WORD_TYPE))
#define SHIFT_BNDM(a, b, n) ((a << (n)) | (b >> (WORD_BITS-(n))))

static void bit_alloc_n(WORD_TYPE **name, int n, int bits) {
   int i;
   name[0] = calloc(n * bit_size(bits), sizeof(WORD_TYPE));
   for (i = 1; i < n; i++) name[i] = name[0] + i*bit_size(bits);
}

/*
 * Backward Nondeterministic DAWG Matching Long patterns
 * The present implementation uses the multiword implementation of the BNDM algorithm
 */

static void bndml_large(search_parameters params)
{
   int i, j;
   WORD_TYPE *D, H;
   WORD_TYPE *B[SIGMA];
   int count = 0;

   bit_alloc_n(B, SIGMA, params.pattern_size);
   for (i = 0; i < params.pattern_size; i++) bit_set(B[params.pattern[params.pattern_size-1-i]], i);

   D = bit_alloc(params.pattern_size);
   j = params.pattern_size-1;
   while (j < params.text_size) {
      int k = 1;
      int l = 0;
      int x = 0;
      for (i = 0; i < bit_size(params.pattern_size); i++) {
         D[i] = B[params.text[j]][i];
         if (D[i]) x = 1;
      }
      while (k < params.pattern_size && x) {
         x = 0;
         if (bit_test(D, params.pattern_size-1)) l = k;
         H = 0;
         for (i = 0; i < bit_size(params.pattern_size); i++) {
            params.pattern_size = D[i];
            D[i] = SHIFT_BNDM(D[i], H, 1) & B[params.text[j-k]][i];
            if (D[i]) x = 1;
            H = params.pattern_size;
         }
         k++;
      }
      if (x) params.match[j-params.pattern_size+1] = 1;
      j += params.pattern_size-l;
   }
   free(D);
   free(B[0]);
}


void bndml(search_parameters params) {
   int B[SIGMA];
   int i, j, s, D, last;

   if (params.pattern_size > 32){
      bndml_large(params);
      return;
   }

   /* Preprocessing */
   for(i=0; i<SIGMA; i++) B[i]=0;
   s=1;
   for (i=params.pattern_size-1; i>=0; i--) {
      B[params.pattern[i]] |= s;
      s <<= 1;
   }

   /* Searching */
   j=0;
   while (j <= params.text_size-params.pattern_size){
      i=params.pattern_size-1; last=params.pattern_size;
      D = ~0;
      while (i>=0 && D!=0) {
         D &= B[params.text[j+i]];
         i--;
         if (D != 0) {
            if (i >= 0) last = i+1;
            else params.match[j-params.pattern_size+1] = 1;
          }
          D <<= 1;
      }
      j += last;
   }
}
