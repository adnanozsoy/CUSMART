#include "ldm.cuh"

__global__
void linear_dawg_matching(unsigned char *text, unsigned long text_size,
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
	       while (L != UNDEFINED && k*pattern_size-1-ell >= 0) {
		     L = getTarget(L, text[k*pattern_size-1-ell]);
		     ++ell;
		     if (L != UNDEFINED && isTerminal(L))
		           R = ell;
	       }
	       while (R > r-k*pattern_size) {
		     if (R == pattern_size) match[k*pattern_size+r-pattern_size] = 1;
		     ++r;
		     if (r == pattern_size) break;
		     R = getSMA(R, text[k*pattern_size-1+r]);
	       }
	 }
	 for (k = (end-1)*pattern_size; k <= text_size - pattern_size && k < boundary; ++k) {
	       for (r = 0; r < pattern_size && pattern[r] == text[r + k]; ++r);
	       if (r >= pattern_size) match[k] = 1;
	 }
}
