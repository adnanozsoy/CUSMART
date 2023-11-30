#ifndef ZT_CUH
#define ZT_CUH

#include "include/define.cuh"

__global__
void zhu_takaoka(unsigned char *text, unsigned long text_size,
		 unsigned char *pattern, int pattern_size,int *bmGs,
		 int **ztBc, int search_len, int *match);

void suffixesZT(unsigned char *x, int m, int *suff);
void preBmGsZT(unsigned char *x, int m, int bmGs[]);
void preZtBcZT(unsigned char *x, int m, int ztBc[SIGMA][SIGMA]);

#endif
