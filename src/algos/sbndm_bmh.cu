
#include "sbndm_bmh.cuh"

#include <stdlib.h>
#include <string.h>


__global__
void simplified_backward_nondeterministic_dawg_horspool_shift(
    unsigned char *text, int text_size,
    unsigned char *pattern, int pattern_size,
    unsigned int *B, int *hbc, int shift, int stride_length, int *match)
{
    int upper_limit;
    int idx = (threadIdx.x + blockIdx.x * blockDim.x) * stride_length;
    if ( idx > text_size - pattern_size ) return;
    if (idx <= text_size - pattern_size - stride_length)
        upper_limit = idx + pattern_size + stride_length;
    else
        upper_limit = text_size;

    /* Searching */
    // if( !memcmp(params.pattern,params.text,params.pattern_size) ) params.match[0] = 1;
    unsigned int D;
    int i = idx + pattern_size;
    while (1) {
        while ((D = B[text[i]]) == 0) i += hbc[text[i+pattern_size]];
        int j = i-1;
        int first = i-pattern_size+1;
        while (1) {
            D = (D << 1) & B[text[j]];
            if (!((j-first) && D)) break;
            j--;
        }
        if (D != 0) {
            if (i >= upper_limit) return;
            match[first] = 1;
            i += shift;
        }
        else {
            i = j+pattern_size;
        }
    }
}

__global__
void simplified_backward_nondeterministic_dawg_horspool_shift_large(
    unsigned char *text, int text_size,
    unsigned char *pattern, int pattern_size,
    unsigned int *B, int *hbc, int shift, int stride_length, int *match)
{

    int upper_limit;
    int idx = (threadIdx.x + blockIdx.x * blockDim.x) * stride_length;
    if ( idx > text_size - pattern_size ) return;
    if (idx <= text_size - pattern_size - stride_length)
        upper_limit = idx + pattern_size + stride_length;
    else
        upper_limit = text_size;

    int p_len = 32;
    int diff = pattern_size-p_len;

    /* Searching */
    //if( !memcmp(pattern,text,pattern_size) ) match[0] = 1;
    unsigned int D;
    int i = idx + p_len;
    while (1) {
        while ((D = B[text[i]]) == 0) i += hbc[text[i+p_len]];
        int j = i-1;
        int first = i-p_len+1;
        while (1) {
            D = (D << 1) & B[text[j]];
            if (!((j-first) && D)) break;
            j--;
        }
        if (D != 0) {
            if (i+diff >= upper_limit) return;
            int k=p_len;
            while (k<pattern_size && pattern[k]==text[first+k]) k++;
            if (k==pattern_size) match[first] = 1;
            i += shift;
        }
        else {
            i = j+p_len;
        }
    }
}
