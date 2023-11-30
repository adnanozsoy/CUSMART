#include "blim.cuh"

__global__
void bit_parallel_length_invariant_matcher(
    unsigned char *text, int text_size,
    unsigned char *pattern, int pattern_size,
    unsigned int *ScanOrder, unsigned int *MScanOrder,
    unsigned long *MM, unsigned int *shift,
    int stride_length, int *match)
{
    unsigned int upper_limit;
    int idx = (threadIdx.x + blockIdx.x*blockDim.x) * stride_length;
    if ( idx > text_size - pattern_size ) return;

    if (idx <= text_size - pattern_size - stride_length)
        upper_limit = idx + stride_length + pattern_size;
    else
        upper_limit = text_size;

    /* Searching */
    unsigned int wsize = WORD - 1 + pattern_size;
    int i = idx;
    unsigned long F = MM[MScanOrder[0]+text[i+ScanOrder[0]]] & 
                      MM[MScanOrder[1]+text[i+ScanOrder[1]]];
    while (i<upper_limit) {
        for (int j=2; F && j<wsize; j++) {
            F &= MM[MScanOrder[j]+text[i+ScanOrder[j]]];
        }
        if (F) {
            for (int j=0; j<WORD; j++)
                if (F & (1<<j))
                    if (i+j<=text_size-pattern_size) match[i+j] = 1;
        }
        i+=shift[text[i+wsize]];
        F = MM[MScanOrder[0]+text[i+ScanOrder[0]]] & 
            MM[MScanOrder[1]+text[i+ScanOrder[1]]];
    }
}
