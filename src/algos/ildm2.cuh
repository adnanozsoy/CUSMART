#ifndef ILDM2_CUH
#define ILDM2_CUH

#include "include/define.cuh"

#define getTarget(p, c) ttrans[(p)*SIGMA+(c)]
#define isTerminal(p) tterminal[(p)]
#define getLength(p) tlength[(p)]

#define setSMA(p, c, q) ttransSMA[(p)*SIGMA+(c)] = (q)
#define getSMA(p, c) ttransSMA[(p)*SIGMA+(c)]

#include <stdlib.h>
#include <string.h>

static void buildSimpleSuffixAutomaton(unsigned char *x, int m, 
   int *ttrans, int *tlength, int *tsuffix, unsigned char *tterminal) {
   int i, art, init, last, p, q, r, counter;
   char c;
  
   init = 0;
   art = 1;
   counter = 2;
   tsuffix[init] = art;
   last = init;
   for (i = 0; i < m; ++i) {
      c = x[i];
      p = last;
      q = counter++;
      tlength[q] = tlength[p] + 1;
      while (p != init && ttrans[SIGMA * p + c] == -1) {
         ttrans[SIGMA * p + c] = q;
         p = tsuffix[p];
      }
      if (ttrans[SIGMA * p + c] == -1) {
         ttrans[SIGMA * init + c] = q;
         tsuffix[q] = init;
      }
      else
         if (tlength[p] + 1 == tlength[ttrans[SIGMA * p + c]]) {
            tsuffix[q] = ttrans[SIGMA * p + c];
         }
         else {
            r = counter++;
            memcpy(ttrans+r*SIGMA, ttrans+ttrans[SIGMA * p + c]*SIGMA, SIGMA*sizeof(int));
            tlength[r] = tlength[p] + 1;
            tsuffix[r] = tsuffix[ttrans[SIGMA * p + c]];
            tsuffix[ttrans[SIGMA * p + c]] = r;
            tsuffix[q] = r;
            while (p != art && tlength[ttrans[SIGMA * p + c]] >= tlength[r]) {
               ttrans[SIGMA * p + c] = r;
               p = tsuffix[p];
            }
         }
      last = q;
   }
   tterminal[last] = 1;
   while (last != init) {
      last = tsuffix[last];
      tterminal[last] = 1;
   }
}


static void preSMA(unsigned char *x, int m, int *ttransSMA) {
   int i, j,state, target, oldTarget;
   unsigned char c;

   memset(ttransSMA, 0, SIGMA*sizeof(int));
   for (state = 0, i = 0; i < m; ++i) {
      oldTarget = getSMA(state, x[i]);
      target = state+1;
      setSMA(state, x[i], target);
      for (j=0, c=0; j < SIGMA; ++c, ++j)
         setSMA(target, c, getSMA(oldTarget, c));
      state = target;
   }
}


__global__
void improved_linear_dawg2(
   unsigned char *text, unsigned long text_size,
   unsigned char *pattern, int pattern_size,
   int *ttrans, int *ttransSMA, unsigned char *tterminal,
   int stride_length, int *match);

#endif
