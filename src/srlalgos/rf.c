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
 * This is an implementation of the Reverse Factor algorithm
 * in T. Lecroq.
 * A variation on the Boyer-Moore algorithm. Theor. Comput. Sci., vol.92, n.1, pp.119--144, (1992).
 */

#include "rf.h"
#include "include/define.h"
#include "include/AUTOMATON.h"

#include "stdio.h"
#include "stdlib.h"
#include "string.h"


void buildSuffixAutomaton(unsigned char *x, int m, int *ttrans, int *tlength, int *tsuffix, unsigned char *tterminal) {
   int i, art, init, last, p, q, r, counter;
   unsigned char c;

   init = 0;
   art = 1;
   counter = 2;
   setSuffixLink(init, art);
   last = init;
   for (i = m-1; i >= 0; --i) {
      c = x[i];
      p = last;
      q = newState();
      setLength(q, getLength(p) + 1);
      while (p != init &&
             getTarget(p, c) == UNDEFINED) {
         setTarget(p, c, q);
         p = getSuffixLink(p);
      }
      if (getTarget(p, c) == UNDEFINED) {
         setTarget(init, c, q);
         setSuffixLink(q, init);
      }
      else
         if (getLength(p) + 1 == getLength(getTarget(p, c)))
            setSuffixLink(q, getTarget(p, c));
         else {
            r = newState();
            //copyVertex(r, getTarget(p, c));
            memcpy(ttrans+r*SIGMA, ttrans+getTarget(p, c)*SIGMA, SIGMA*sizeof(int));
            setSuffixLink(r, getSuffixLink(getTarget(p, c)));
            setLength(r, getLength(p) + 1);
            setSuffixLink(getTarget(p, c), r);
            setSuffixLink(q, r);
            while (p != art && getLength(getTarget(p, c)) >= getLength(r)) {
               setTarget(p, c, r);
               p = getSuffixLink(p);
            }
         }
      last = q;
   }
   setTerminal(last);
   while (last != init) {
      last = getSuffixLink(last);
      setTerminal(last);
   }
}


void rf(search_parameters params) {

   int i, j, shift, period, init, state;
   int *ttrans, *tlength, *tsuffix;
   unsigned char *tterminal;
   int mMinus1, nMinusm, size;
 
   /* Preprocessing */
   mMinus1 = params.pattern_size - 1;
   nMinusm = params.text_size - params.pattern_size;
   size = 2 * params.pattern_size + 3;
   ttrans = (int *)malloc(size*SIGMA*sizeof(int));
   tlength = (int *)calloc(size, sizeof(int));
   tsuffix = (int *)calloc(size, sizeof(int));
   tterminal = (unsigned char *)calloc(size, sizeof(unsigned char));
   memset(ttrans, -1, (2 * params.pattern_size + 3)*SIGMA*sizeof(int));
   buildSuffixAutomaton(params.pattern, params.pattern_size, ttrans, tlength, tsuffix, tterminal);
   init = 0;
   period = params.pattern_size;
 
   /* Searching */
  
   if (strncmp(params.pattern, params.text, params.pattern_size) == 0)
      params.match[0] = 1;
   j = 1;
   while (j <= nMinusm) {
      i = mMinus1;
      state = init;
      shift = params.pattern_size;
      while (getTarget(state, params.text[i + j]) != UNDEFINED) {
         state = getTarget(state, params.text[i + j]);
         if (isTerminal(state)) {
            period = shift;
            shift = i;
         }
         --i;
      }
      if (i < 0) {
         params.match[j] = 1;
         shift = period;
      }
      j += shift;
   }
   
   free(ttrans);
   free(tlength);
   free(tsuffix);
   free(tterminal);   
}

