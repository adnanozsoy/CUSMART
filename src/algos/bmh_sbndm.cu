
#include "bmh_sbndm.cuh"

__global__
void horspool_with_bndm_large(
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
    //if( !memcmp(pattern,text,p_len) ) match[0] = 1;
    int i = idx + p_len;
    while (i+diff < upper_limit) {
        int k;
        while ( (k=hbc[text[i]])!=0 ) i+=k;
        int j=i;
        int s=i-p_len+1;
        unsigned int D = B[text[j]];
        while (D!=0) {
            j--;
            D = (D<<1) & B[text[j]];
        }
        if (j<s) {
            if (s<upper_limit) {
                k = p_len;
                while (k<pattern_size && pattern[k]==text[s+k]) k++;
                if (k==pattern_size && i+diff<upper_limit) match[s] = 1;
            }
            i += shift;
        }
        else i = j+p_len;
    }
}


__global__
void horspool_with_bndm(
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
    //if( !memcmp(pattern,text,pattern_size) ) match[0] = 1;
    int i = idx + pattern_size;
    while (i < upper_limit) {
        int k;
        while ( (k=hbc[text[i]])!=0 ) i+=k;
        int j=i;
        int s = i-pattern_size+1;
        unsigned int D = B[text[j]];
        while (D!=0) {
            j--;
            D = (D<<1) & B[text[j]];
        }
        if (j<s) {
            if (s<upper_limit && i<upper_limit) match[s] = 1;
            i += shift;
        }
        else i = j+pattern_size;
    }
}

/*
 * Horspool algorithm with BNDM test designed for large patterns
 * The present implementation searches for prefixes of the pattern of length 32.
 * When an occurrence is found the algorithm tests for the whole occurrence of the pattern
 */
