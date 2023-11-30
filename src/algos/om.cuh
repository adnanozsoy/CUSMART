#ifndef OM_CUH
#define OM_CUH

#include "include/define.cuh"

#define MAX(a,b) ((a) > (b) ? (a) : (b))

typedef struct patternScanOrder { 
    int loc; 
    unsigned char c; 
} ompattern; 

__global__ 
void optimal_mismatch(unsigned char *text, unsigned long text_size,
	unsigned char *pattern, int pattern_size,int *adaptedGs, int *qsBc, 
	ompattern *pat, int search_len, int *match);


void om_preprocess(unsigned char *x, int m, ompattern *pat, int qsBc[], int adaptedGs[]);

#endif


