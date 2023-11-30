#include "dfdm.cuh"

__global__
void double_forward_dawg( 
        unsigned char *text, int text_size, 
        unsigned char *pattern, int pattern_size,
        int *ttrans, int *tlength, int *tsuffix, int alpha, int beta, int logM,
        int stride_length, int *match)
{
    int j, end;

    int upper_limit;
    int idx = (threadIdx.x + blockIdx.x*blockDim.x) * stride_length;
    if ( idx > text_size - pattern_size ) return;

    if (idx <= text_size - pattern_size - stride_length)
        upper_limit = idx + stride_length;
    else
        upper_limit = text_size - pattern_size;


    /* Searching */
    int ell = 0;
    int init = 0;
    int state2 = init;
    int s = idx;
    while (s <= upper_limit) {
        int state1 = init;
        j = s+beta;
        end = s+pattern_size;
        while ((j < end) && (ttrans[SIGMA * state1 + text[j]] != -1)) {
            state1 = ttrans[SIGMA * state1 + text[j]];
            ++j;
        }

        if (j < s+pattern_size) {
            state2 = tsuffix[state1];
            while ((state2 != init) && (ttrans[SIGMA * state2 + text[j]] == -1)) {
                state2 = tsuffix[state2];
            }
            if (ttrans[SIGMA * state2 + text[j]] != -1) {
                ell = tlength[state2] + 1;
                state2 = ttrans[SIGMA * state2 + text[j]];
            }
            else {
                ell = 0;
                state2 = init;
            }
            ++j;
        }
        else {
            j = s+ell;
        }
        end = s+pattern_size;
        while (((j < end) || (ell < alpha)) && (j < text_size)) {
            if (ttrans[SIGMA * state2 + text[j]] != -1) {
                ++ell;
                state2 = ttrans[SIGMA * state2 + text[j]];
            }
            else {
                while ((state2 != init) && (ttrans[SIGMA * state2 + text[j]] == -1)) {
                    state2 = tsuffix[state2];
                }
                if (ttrans[SIGMA * state2 + text[j]] != -1) {
                    ell = tlength[state2] + 1;
                    state2 = ttrans[SIGMA * state2 + text[j]];
                }
                else {
                    ell = 0;
                    state2 = init;
                }
            }
            if (ell == pattern_size) {
                match[j-pattern_size+1] = 1;
                state2 = tsuffix[state2];
                ell = tlength[state2];
            }
            ++j;
        }
        s = j-ell;
    }
}


