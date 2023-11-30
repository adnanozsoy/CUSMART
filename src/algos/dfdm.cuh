#ifndef DFDM_CUH
#define DFDM_CUH

#include "include/define.cuh"

#include <stdlib.h>
#include <string.h>


static void buildSimpleSuffixAutomatonFDM(unsigned char *x, int m, 
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


__global__
void double_forward_dawg( 
         unsigned char *text, int text_size, 
         unsigned char *pattern, int pattern_size,
         int *ttrans, int *tlength, int *tsuffix, int alpha, int beta,  int logM,
         int stride_length, int *match);

#endif
