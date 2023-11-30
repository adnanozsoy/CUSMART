
#include "ildm1.cuh"

__global__
void improved_linear_dawg1(
    unsigned char *text, unsigned long text_size,
    unsigned char *pattern, int pattern_size,
    int *ttrans, int *ttransSMA, unsigned char *tterminal,
    int stride_length, int *match)
{

    int upper_limit;
    int idx = (threadIdx.x + blockIdx.x*blockDim.x) * stride_length;
    if ( idx > text_size - pattern_size ) return;

    if (idx <= text_size - pattern_size - stride_length)
        upper_limit = idx + pattern_size + stride_length;
    else
        upper_limit = text_size;

    /* Searching */
    int k = idx + pattern_size-1;
    while ( k < upper_limit ) {
        int L = 0;
        int R = 0;
        int l = 0;
        while ( k-l >= 0 && ( L = getTarget(L, text[k-l]) ) != -1 ) {
            l++;
            if ( isTerminal(L) ) R = l;
        }
        while ( R > 0 ) {
            if ( R==pattern_size )
                match[k-pattern_size+1] = 1;
            k++;
            if ( k >= upper_limit ) break;
            R = getSMA(R, text[k]);
        }
        k = k + pattern_size;
    }
}

