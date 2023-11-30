#include "ww.cuh"

__global__
void wide_window(unsigned char *text, unsigned long text_size,
		 unsigned char *pattern, int pattern_size,
		 int *ttrans, int *ttransSMA, unsigned char *tterminal,
		 int search_len, int *match){
         int k, R, L, r, ell, end;
	 unsigned long thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	 unsigned long start_inx = thread_id * search_len;    
	 unsigned long boundary = start_inx + search_len + pattern_size  - 1;
	 boundary = boundary > text_size ? text_size : boundary;

      	 end = text_size/pattern_size;
	 if (text_size%pattern_size > 0) ++end;
	 for (k = start_inx; k < end && k < boundary; ++k) {
	       R = L = r = ell = 0;
	       while (R != UNDEFINED && k*pattern_size-1+r < text_size) {
		     R = getTarget(R, text[k*pattern_size-1+r]);
		     ++r;
		     if (R != UNDEFINED && isTerminal(R))
		           L = r;
	       }
	       while (L > ell) {
		     if (L == pattern_size) match[k*pattern_size-1-ell] = 1;
		     ++ell;
		     if (ell == pattern_size)
		           break;
		     L = getSMA(L, text[k*pattern_size-1-ell]);
	       }
	 }
	 for (k = (end-1)*pattern_size; k <= text_size - pattern_size && k < boundary; ++k) {
	       for (r = 0; r < pattern_size && pattern[r] == text[r + k]; ++r);
	           if (r >= pattern_size) match[k] = 1;
	 }
}

void preSMARev(unsigned char *x, int m, int *ttransSMA) {
   int c, i, state, target, oldTarget;

   memset(ttransSMA, 0, SIGMA*sizeof(int));
   for (state = 0, i = m-1; i >= 0; --i) {
      oldTarget = getSMA(state, x[i]);
      target = state+1;
      setSMA(state, x[i], target);
      for (c = 0; c < SIGMA; ++c)
         setSMA(target, c, getSMA(oldTarget, c));
      state = target;
   }
}
