#include "ms.cuh"

#include <stdlib.h>

//int minShift[XSIZE];
int *minShift;

__global__ void maximal_shift(unsigned char *text, int text_size, unsigned char *pattern, unsigned long pattern_size,
			      int *qsBc, int *adaptedGs, patternS *pat, int search_len, int *match) {

        unsigned long thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned long start_inx = thread_id * search_len;

	unsigned long boundary = start_inx + search_len + pattern_size - 1;
	boundary = boundary > text_size ? text_size : boundary;
	int i;
	unsigned long j = start_inx;
  
	/* Searching */
	while (j < boundary && j <= text_size - pattern_size) {
	        i = 0;
	        while (i < pattern_size && pat[i].c == text[j + pat[i].loc])  ++i;
		if (i >= pattern_size) match[j] = 1;
		int a = adaptedGs[i];
		int b = qsBc[text[j + pattern_size]];
		j += ((a) > (b) ? (a) : (b));
	}
}

void computeMinShift(unsigned char *x, int m) {
  int i, j;
  minShift = (int *)malloc((m+1) * sizeof(int));
  for (i = 0; i < m; ++i) {
    for (j = i - 1; j >= 0; --j)
      if (x[i] == x[j])
	break;
    minShift[i] = i - j;
  }
}

void orderPatternMS(unsigned char *x, int m, int (*pcmp)(const void*, const void*), patternS *pat) {
  int i;
  for (i = 0; i < m; ++i) {
    pat[i].loc = i;
    pat[i].c = x[i];
  }
  qsort(pat, m, sizeof(patternS), pcmp);
}

void preQsBcMS(unsigned char *x, int m, int qbc[]) {
  int i;
  for (i=0;i<SIGMA;i++)   qbc[i]=m+1;
  for (i=0;i<m;i++) qbc[x[i]]=m-i;
}

int matchShiftMS(unsigned char *x, int m, int ploc, int lshift, patternS *pat) {
  int i, j;
  for (; lshift < m; ++lshift) {
    i = ploc;
    while (--i >= 0) {
      if ((j = (pat[i].loc - lshift)) < 0)  continue;
      if (pat[i].c != x[j]) break;
    }
    if (i < 0) break;
  }
  return(lshift);
}

void preAdaptedGsMS(unsigned char *x, int m, int adaptedGs[], patternS *pat) {
  int lshift, i, ploc;

  adaptedGs[0] = lshift = 1;
  for (ploc = 1; ploc <= m; ++ploc) {
    lshift = matchShiftMS(x, m, ploc, lshift, pat);
    adaptedGs[ploc] = lshift;
  }
  for (ploc = 0; ploc <= m; ++ploc) {
    lshift = adaptedGs[ploc];
    while (lshift < m) {
      i = pat[ploc].loc - lshift;
      if (i < 0 || pat[ploc].c != x[i])
	break;
      ++lshift;
      lshift = matchShiftMS(x, m, ploc, lshift, pat);
    }
    adaptedGs[ploc] = lshift;
  }
}


int maxShiftPcmp(const void* pa, const void* pb) {
  int dsh;
  patternS* pat1 = (patternS*)pa;
  patternS* pat2 = (patternS*)pb;
  dsh = minShift[pat2->loc] - minShift[pat1->loc];
  return(dsh ? dsh : (pat2->loc - pat1->loc));
}
