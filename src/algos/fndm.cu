
#include "fndm.cuh"

__global__
void forward_nondeterministic_dawg(
    unsigned char *text, int text_size,
    unsigned char *pattern, int pattern_size,
    int *B, int stride_length, int *match)
{
    int upper_limit;
    int idx = (threadIdx.x + blockIdx.x*blockDim.x) * stride_length;
    if ( idx > text_size - pattern_size ) return;
    if (idx <= text_size - pattern_size - stride_length)
        upper_limit = idx + pattern_size + stride_length;
    else
        upper_limit = text_size;

    unsigned int NEG0 = ~0;
    unsigned int NEG0m1 = ~0<<(pattern_size-1);

    /* Searching */
    int i = idx + pattern_size - 1;
    while ( i < upper_limit ) {
        unsigned int D = B[text[i]];
        while (D != NEG0) {
            if (D < NEG0m1) {
                int k = 0;
                int first=i-pattern_size+1;
                while (k<pattern_size && pattern[k]==text[first+k]) k++;
                if (k==pattern_size && i<text_size) match[first] = 1;
            }
            i = i+1;
            D = (D<<1) | B[text[i]];
        }
        i=i+pattern_size;
    }
}

/*
 * Forward Nondeterministic DAWG Matching algorithm designed for large patterns
 * The present implementation searches for prefixes of the pattern of length 32.
 * When an occurrence is found the algorithm tests for the whole occurrence of the pattern
 */

__global__
void forward_nondeterministic_dawg_large(
    unsigned char *text, int text_size,
    unsigned char *pattern, int pattern_size,
    int *B, int stride_length, int *match)
{
    int upper_limit;
    int idx = (threadIdx.x + blockIdx.x*blockDim.x) * stride_length;
    if ( idx > text_size - pattern_size ) return;
    if (idx <= text_size - pattern_size - stride_length)
        upper_limit = idx + pattern_size + stride_length;
    else
        upper_limit = text_size;

    int p_len = 32;
    unsigned int NEG0 = ~0;
    unsigned int NEG0m1 = ~0<<(p_len-1);

    /* searching */
    int i = idx + p_len - 1;
    while ( i < upper_limit ) {
        unsigned int D = B[text[i]];
        while (D != NEG0) {
            if (D < NEG0m1) {
                int k = 0;
                int first=i-p_len+1;
                while (k<pattern_size && pattern[k]==text[first+k]) k++;
                if (k==pattern_size && i<text_size) match[first] = 1;
            }
            i = i+1;
            D = (D<<1) | B[text[i]];
        }
        i=i+p_len;
    }
}
