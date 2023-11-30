#include "sabp.cuh"

__device__
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

__device__
int mylog2(int unsigned n) {
   int ell;
   ell = 0;
   while (n >= 2) {
      ++ell;
      n /= 2;
   }
   return ell;
}

__global__
void small_alphabet_bit_parallel_large(unsigned char *text, int text_size,
				      unsigned char *pattern, int pattern_size,
				      unsigned int *T, unsigned int mask, unsigned int mask2, int search_len, int *match) {
        int k, first, p_len, m;
        unsigned int b, D, Delta;
        p_len = pattern_size;
        m = 30;

	unsigned long thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned long start_inx = thread_id * search_len;
	unsigned long boundary = start_inx + search_len + m - 1;
	boundary = boundary > text_size ? text_size : boundary;
	unsigned long i = start_inx + m - 1;
	unsigned long j;
	
	D = 0;
	b = mask;
	j = i;
	while (i < boundary && i < text_size) {
	  D |= (T[text[j]] << (i - j));
	  D &= mask;
	  b &= ~pow2(m - i + j - 1);
	  if ((D & mask2) == 0) {
	    if (b == 0) {
	      D |= mask2;
	      k=0;
	      first = i-m+1;
	      while(k<p_len && pattern[k]==text[first+k]) k++;
	      if (k==p_len) match[j] = 1;
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

__global__
void small_alphabet_bit_parallel(unsigned char *text, int text_size,
				 unsigned char *pattern, int pattern_size,
				 unsigned int *T, unsigned int mask, unsigned int mask2, int search_len, int *match) {
	unsigned int b, D, Delta;
        unsigned long thread_id = blockDim.x * blockIdx.x + threadIdx.x;
        unsigned long start_inx = thread_id * search_len;
        unsigned long boundary = start_inx + search_len + pattern_size - 1;
        boundary = boundary > text_size ? text_size : boundary;
        unsigned long i = start_inx + pattern_size - 1;
        unsigned long j;

	D = 0;
	b = mask;
	j = i;
	while (i < boundary && i < text_size) {
	  D |= (T[text[j]] << (i - j));
	  D &= mask;
	  b &= ~pow2(pattern_size - i + j - 1);
	  if ((D & mask2) == 0) {
	    if (b == 0) {
	      D |= mask2;
	      match[i - pattern_size + 1] = 1;
	    }
	    else {
	      j = i - (pattern_size - mylog2(b) - 1);
	      continue;
	    }
	  }
	  if (D == mask) {
	    D = 0;
	    b = mask;
	    i += pattern_size;
	  }
	  else {
	    Delta = pattern_size - mylog2(~D & mask) - 1;
	    D <<= Delta;
	    b = ((b | ~mask) >> Delta) & mask;
	    i += Delta;
	  }
	  j = i;
	}
}
