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
 * This is an implementation of the Turbo Reverse Factor algorithm
 * in M. Crochemore and A. Czumaj and L. Gcasieniec and S. Jarominek and T. Lecroq and W. Plandowski and W. Rytter.
 * Speeding up two string matching algorithms. Algorithmica, vol.12, n.4/5, pp.247--267, (1994).
 */

#include "trf.h"
#include "include/define.h"
#include "include/AUTOMATON.h"

#include "stdlib.h"
#include "string.h"

void preMpforTRF(unsigned char *x, int m, int mpNext[]) {
   int i, j;
   i = 0;
   j = mpNext[0] = -1;
   while (i < m) {
      while (j > -1 && x[i] != x[j])
         j = mpNext[j];
      mpNext[++i] = ++j;
   }
}

void buildSuffixAutomaton4TRF(unsigned char *x, int m, 
   int *ttrans, int *tlength, int *tposition, int *tsuffix, unsigned char *tterminal, int *tshift) {
   int i, art, init, last, p, q, r, counter, tmp;
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
      setPosition(q, getPosition(p) + 1);
      while (p != init &&
             getTarget(p, c) == UNDEFINED) {
         setTarget(p, c, q);
         setShift(p, c, getPosition(q) - getPosition(p) - 1);
         p = getSuffixLink(p);
      }
      if (getTarget(p, c) == UNDEFINED) {
         setTarget(init, c, q);
         setShift(init, c, getPosition(q) - getPosition(init) - 1);
         setSuffixLink(q, init);
      }
      else
         if (getLength(p) + 1 == getLength(getTarget(p, c)))
            setSuffixLink(q, getTarget(p, c));
         else {
            r = newState();
            tmp = getTarget(p, c);
            memcpy(ttrans+r*SIGMA, ttrans+tmp*SIGMA, SIGMA*sizeof(int));
            memcpy(tshift+r*SIGMA, tshift+tmp*SIGMA, SIGMA*sizeof(int));
            setPosition(r, getPosition(tmp));
            setSuffixLink(r, getSuffixLink(tmp));
            setLength(r, getLength(p) + 1);
            setSuffixLink(tmp, r);
            setSuffixLink(q, r);
            while (p != art && getLength(getTarget(p, c)) >= getLength(r)) {
               setShift(p, c, getPosition(getTarget(p, c)) - getPosition(p) - 1);
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



void trf(search_parameters params) {
   int period, i, j, shift, u, periodOfU, disp, init, count, state, mu, *mpNext;
   int *ttrans, *tlength, *tposition, *tsuffix, *tshift;
   unsigned char *tterminal;
   int mMinus1, nMinusm, size;
  
   /* Preprocessing */
   nMinusm = params.text_size-params.pattern_size;
   mMinus1 = params.pattern_size-1;
   size = 2*params.pattern_size+3;
   mpNext = (int *)malloc((params.pattern_size+1)*sizeof(int));
   ttrans = (int *)malloc(size*SIGMA*sizeof(int));
   tshift = (int *)malloc(size*SIGMA*sizeof(int));
   tlength = (int *)calloc(size, sizeof(int));
   tposition = (int *)calloc(size, sizeof(int));
   tsuffix = (int *)calloc(size, sizeof(int));
   tterminal = (unsigned char *)calloc(size, sizeof(unsigned char));
   memset(ttrans, -1, (2*params.pattern_size+3)*SIGMA*sizeof(int));
   buildSuffixAutomaton4TRF(params.pattern, params.pattern_size, ttrans, tlength, tposition, tsuffix, tterminal, tshift);
   init = 0;
   preMpforTRF(params.pattern, params.pattern_size, mpNext);
   period = params.pattern_size - mpNext[params.pattern_size];
   i = 0;
   shift = params.pattern_size;
  
   /* Searching */
   count = 0;
   if (strncmp(params.pattern, params.text, params.pattern_size) == 0)
     params.match[0] = 1;
   j = 1;
   while (j <= nMinusm) {
      i = mMinus1;
      state = init;
      u = mMinus1 - shift;
      periodOfU = (shift != params.pattern_size ?  params.pattern_size - shift - mpNext[params.pattern_size - shift] : 0);
      shift = params.pattern_size;
      disp = 0;
      while (i > u && getTarget(state, params.text[i + j]) != UNDEFINED) {
         disp += getShift(state, params.text[i + j]);
         state = getTarget(state, params.text[i + j]);
         if (isTerminal(state))
            shift = i;
         --i;
      }
      if (i <= u)
         if (disp == 0) {
            params.match[j] = 1;
            shift = period;
         }
         else {
            mu = (u + 1)/2;
            if (periodOfU <= mu) {
               u -= periodOfU;
               while (i > u && getTarget(state, params.text[i + j]) != UNDEFINED) {
                  disp += getShift(state, params.text[i + j]);
                  state = getTarget(state, params.text[i + j]);
                  if (isTerminal(state))
                     shift = i;
                  --i;
               }
               if (i <= u)
                  shift = disp;
            }
            else {
               u = u - mu - 1;
               while (i > u && getTarget(state, params.text[i + j]) != UNDEFINED) {
                  disp += getShift(state, params.text[i + j]);
                  state = getTarget(state, params.text[i + j]);
                  if (isTerminal(state))
                     shift = i;
                  --i;
               }
            }
         }
      j += shift;
   }
   free(mpNext);
   free(ttrans);
   free(tshift);
   free(tlength);
   free(tposition);
   free(tsuffix);
   free(tterminal);
}
