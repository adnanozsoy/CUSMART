#include "bndm.cuh"

__global__
void backward_nondeterministic_dawg( 
    unsigned char *text, int text_size, 
    unsigned char *pattern, int pattern_size,
    int *B, int stride_length, int *match)
{
    int upper_limit;
    int idx = (threadIdx.x + blockIdx.x*blockDim.x) * stride_length;
    if ( idx > text_size - pattern_size ) return;

    if (idx <= text_size - pattern_size - stride_length)
        upper_limit = idx + stride_length;
    else
        upper_limit = text_size - pattern_size;


    /* Searching */
    int j = idx;
    while (j <= upper_limit) {
        int i = pattern_size - 1;
        int last = pattern_size;
        int D = ~0;
        while (i>=0 && D!=0) {
            D &= B[text[j+i]];
            i--;
            if (D != 0) {
                if (i >= 0) last = i+1;
                else match[j] = 1;
            }
            D <<= 1;
        }
        j += last;
    }
}

__global__
void backward_nondeterministic_dawg_large(
    unsigned char *text, int text_size, 
    unsigned char *pattern, int pattern_size,
    int *B, int stride_length, int *match)
{
    int upper_limit;
    int idx = (threadIdx.x + blockIdx.x*blockDim.x) * stride_length;
    if ( idx > text_size - pattern_size ) return;

    if (idx <= text_size - pattern_size - stride_length)
        upper_limit = idx + stride_length;
    else
        upper_limit = text_size - pattern_size;

    int p_len = 32;

    /* Searching */
    int j = idx;
    while (j <= upper_limit) {
        int i = p_len - 1;
        int last = p_len;
        int D = ~0;
        while (i>=0 && D!=0) {
            D &= B[text[j+i]];
            i--;
            if (D != 0) {
                if (i >= 0)
                    last = i+1;
                else {
                    int k = p_len;
                    while (k<pattern_size && pattern[k]==text[j+k]) k++;
                    if (k==pattern_size) match[j] = 1;
                }
            }
            D <<= 1;
        }
        j += last;
    }
}


