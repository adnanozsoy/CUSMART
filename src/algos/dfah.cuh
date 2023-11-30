#ifndef DFAH_CUH
#define DFAH_CUH

#include "include/define.cuh"

#define setSMA(p, c, q) ttransSMA[(p)*SIGMA+(c)] = (q)
#define getSMA(p, c) ttransSMA[(p)*SIGMA+(c)]

__global__ 
void high_deterministic_finite_automaton(unsigned char *text, int text_size, 
                            unsigned char *pattern, int pattern_size,int *ttransSMA, 
                            int *match);
__host__
void preHSMA(unsigned char *x, int m, int *ttransSMA);
#endif
