#ifndef MS_CUH
#define MS_CUH

#include "include/define.cuh"

typedef struct patternScanOrder {
  int loc;
  char c;
} patternS;  

__global__
void maximal_shift(unsigned char *pattern, int pattern_size, unsigned char *text, unsigned long text_size,
		   int *qsBc, int *adaptedGs, patternS *pat, int search_len, int *match);
int maxShiftPcmp(const void* pa, const void* pb);
void preAdaptedGsMS(unsigned char *x, int m, int adaptedGs[], patternS *pat);
int matchShiftMS(unsigned char *x, int m, int ploc, int lshift, patternS *pat);
void preQsBcMS(unsigned char *x, int m, int qbc[]);
void orderPatternMS(unsigned char *x, int m, int (*pcmp)(const void*, const void*), patternS *pat);
void computeMinShift(unsigned char *x, int m);

#endif
