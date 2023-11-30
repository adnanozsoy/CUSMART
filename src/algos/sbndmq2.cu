#include "sbndmq2.cuh"

#include <stdlib.h>
#include <string.h>

#define GRAM2(B, y, j) (B[y[j]]<<1)&B[y[j-1]]

__global__
void simplified_backward_nondeterministic_dawg_qgram(
    unsigned char *text, int text_size,
    unsigned char *pattern, int pattern_size,
    unsigned int *B, int mq, int mMinusq, int shift,
    int stride_length, int *match)
{
    const unsigned int q = 2;
    int upper_limit;
    int idx = (threadIdx.x + blockIdx.x * blockDim.x) * stride_length;
    if ( idx > text_size - pattern_size ) return;
    if (idx <= text_size - pattern_size - stride_length)
        upper_limit = idx + pattern_size + stride_length;
    else
        upper_limit = text_size;

    /* Searching */
    //if( !memcmp(pattern,text,pattern_size) ) match[0] = 1;
    int j = idx + pattern_size;
    while (j < upper_limit) {
        unsigned int D = GRAM2(B, text, j);
        if (D != 0) {
            int pos = j;
            while (D=(D<<1) & B[text[j-q]]) --j;
            j += mq;
            if (j == pos) {
                match[j] = 1;
                j+=shift;
            }
        }
        else j+=mMinusq;
    }
}

__global__
void simplified_backward_nondeterministic_dawg_qgram_large(
    unsigned char *text, int text_size,
    unsigned char *pattern, int pattern_size,
    unsigned int *B, int mq, int mMinusq, int shift,
    int stride_length, int *match)
{
    const unsigned int q = 2;
    const int p_len = 32;

    int upper_limit;
    int idx = (threadIdx.x + blockIdx.x * blockDim.x) * stride_length;
    if ( idx > text_size - pattern_size ) return;
    if (idx <= text_size - pattern_size - stride_length)
        upper_limit = idx + pattern_size + stride_length;
    else
        upper_limit = text_size;

    /* Searching */
    // if( !memcmp(pattern,text,pattern_size) ) match[0] = 1;
    int j = idx + p_len;
    while (j < upper_limit) {
        unsigned int D = GRAM2(B, text, j);
        if (D != 0) {
            int pos = j;
            while (D=(D<<1) & B[text[j-q]]) --j;
            j += mq;
            if (j == pos) {
                int i;
                for (i=p_len+1; i<pattern_size && pattern[i]==text[j-p_len+1+i]; i++);
                if (i==pattern_size) match[j-p_len+1] = 1;
                j+=shift;
            }
        }
        else j+=mMinusq;
    }
}



