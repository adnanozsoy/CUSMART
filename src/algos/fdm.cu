#include "fdm.cuh"

__global__
void forward_dawg( 
        unsigned char *text, int text_size, 
        unsigned char *pattern, int pattern_size,
        int *ttrans, int *tlength, int *tsuffix, 
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
    int ell = 0;
    int init = 0;
    int state = init;
    for (int j = idx; j < upper_limit; ++j) {
        if (ttrans[SIGMA * state + text[j]] != -1) {
            ++ell;
            state = ttrans[SIGMA * state + text[j]];
        }
        else {
            while (state != init && ttrans[SIGMA * state + text[j]] == -1)
                state = tsuffix[state];
            if (ttrans[SIGMA * state + text[j]] != -1) {
                ell = tlength[state] + 1;
                state = ttrans[SIGMA * state + text[j]];
            }
            else {
                ell = 0;
                state = init;
            }
        }
        if (ell == pattern_size)
            match[j - pattern_size + 1] = 1;
    }
}


