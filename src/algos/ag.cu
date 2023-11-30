
#include "ag.cuh"

#include "stdio.h"
#include "stdlib.h"
#include "string.h"

#define MAX(a,b) ((a) > (b) ? (a) : (b))

__host__
void preBmBcAG(unsigned char *pattern, int pattern_size, int *bmBc) {
    int i;
    for (i = 0; i < SIGMA; ++i) bmBc[i] = pattern_size;
    for (i = 0; i < pattern_size - 1; ++i) bmBc[pattern[i]] = pattern_size - i - 1;
}
 
 __host__
void suffixesAG(unsigned char *pattern, int pattern_size, int *suff) {
    int f = 0, g, i;
    suff[pattern_size - 1] = pattern_size;
    g = pattern_size - 1;
    for (i = pattern_size - 2; i >= 0; --i) {
        if (i > g && suff[i + pattern_size - 1 - f] < i - g)
            suff[i] = suff[i + pattern_size - 1 - f];
        else {
            if (i < g) g = i;
            f = i;
            while (g >= 0 && pattern[g] == pattern[g + pattern_size - 1 - f]) --g;
            suff[i] = f - g;
        }
    }
}

__host__
void preBmGsAG(unsigned char *pattern, int pattern_size, int *bmGs, int *suff) {
    int i, j;
    for (i = 0; i < pattern_size; ++i) bmGs[i] = pattern_size;
    j = 0;
    for (i = pattern_size - 1; i >= 0; --i)
        if (suff[i] == i + 1)
            for (; j < pattern_size - 1 - i; ++j)
                if (bmGs[j] == pattern_size)
                   bmGs[j] = pattern_size - 1 - i;
    for (i = 0; i <= pattern_size - 2; ++i)
        bmGs[pattern_size - 1 - suff[i]] = pattern_size - 1 - i;
}

__global__
void ag(unsigned char *text, int text_size, 
        unsigned char *pattern, int pattern_size,
        int *bmGs, int *bmBc, int *suff,
        int stride_length, int *match) {

    int i, j, k, s, shift;
    int upper_limit;
    int skip[20];

    unsigned long idx = (threadIdx.x + blockIdx.x*blockDim.x) * stride_length;
    if ( idx > text_size - pattern_size ) return;

    if (idx <= text_size - pattern_size - stride_length)
        upper_limit = stride_length + idx;
    else
        upper_limit = text_size - pattern_size;

    /* Searching */
    j = idx;
   	while (j <= upper_limit) {
        i = pattern_size - 1;
        while (i >= 0) {
            k = skip[i];
            s = suff[i];
            if (k > 0)
                if (k > s) {
                    if (i + 1 == s) 
                        i = -1;
                    else 
                        i -= s;
                    break;
                }
                else {
                    i -= k;
                    if (k < s) break;
                }
            else {
                if (pattern[i] == text[i + j]) --i;
                else break;
            }
        }
        if (i < 0) {
            match[j] = 1;
            skip[pattern_size - 1] = pattern_size;
            shift = bmGs[0];
        }
        else {
            skip[pattern_size - 1] = pattern_size - 1 - i;
            shift = MAX(bmGs[i], bmBc[text[i + j]] - pattern_size + 1 + i);
        }
        j += shift;
        memcpy(skip, skip + shift, (pattern_size - shift)*sizeof(int));
        memset(skip + pattern_size - shift, 0, shift*sizeof(int));
    }
}
