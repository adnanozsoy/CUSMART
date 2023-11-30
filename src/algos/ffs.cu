
#include "ffs.cuh"
#include <stdlib.h>
#include <string.h>

__host__
void forward_suffix_function(unsigned char *x, int m, int *bm_gs, int alpha)
{
    int init;
    int i, j, last, suffix_len;
    int *temx = (int*)malloc(m*sizeof(int));
    init = 0;
    for (i=0; i<m; i++) for (j=init; j<init+alpha; j++) bm_gs[SIGMA*i+j] = m+1;
    for (i=0; i<m; i++) temx[i]=i-1;
    for (suffix_len=0; suffix_len<=m; suffix_len++) {
        last = m-1;
        i = temx[last];
        while (i>=0) {
            if ( bm_gs[SIGMA*(m-suffix_len)+x[i+1]]>m-1-i )
                if (i-suffix_len<0 || (i-suffix_len>=0 && x[i-suffix_len]!=x[m-1-suffix_len]))
                    bm_gs[SIGMA*(m-suffix_len)+x[i+1]]=m-1-i;
            if ((i-suffix_len >= 0 && x[i-suffix_len]==x[last-suffix_len])
                || (i-suffix_len <  0)) {
                temx[last]=i;
                last=i;
            }
            i = temx[i];
        }
        if (bm_gs[SIGMA*(m-suffix_len)+x[0]] > m) bm_gs[SIGMA*(m-suffix_len)+x[0]] = m;
        temx[last]=-1;
    }
    free(temx);
}

__global__
void forward_fast_search(
    unsigned char *text, int text_size,
    unsigned char *pattern, int pattern_size,
    int *bc, int *gs, int stride_length, int *match)
{
    int j, k;
    int upper_limit;
    int idx = (threadIdx.x + blockIdx.x * blockDim.x) * stride_length;
    if ( idx > text_size - pattern_size ) return;
    if (idx <= text_size - pattern_size - stride_length)
        upper_limit = idx + pattern_size + stride_length;
    else
        upper_limit = text_size;

    /* Searching */
    // if( !memcmp(params.pattern,params.text,params.pattern_size) ) count++;
    int s = idx + pattern_size;
    while (s<upper_limit) {
        while (k=bc[text[s]])   s += k;
        for (j=s-1, k=pattern_size-1; k>0 && pattern[k-1]==text[j]; k--, j--);
            if (!k && s<text_size) match[j] = 1;
        s += gs[SIGMA*k+text[s+1]];
    }
}



