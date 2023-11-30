
#include "aoso2.cuh"

__device__
static void aoso2_verify(
    unsigned char *text, int text_size,
    unsigned char *pattern, int pattern_size,
    int *match, int j, int q, unsigned int D, unsigned int mm)
{
    unsigned int s;
    int i, c, k;
    D = (D & mm)^mm;
    while (D != 0) {
        s = LOG2(D);
        c = -(pattern_size/q-1)*q-s/(pattern_size/q);
        k = 0;
        i = j+c;
        if (i>=0 && i<=text_size-pattern_size)
            while (k<pattern_size && pattern[k]==text[i+k]) k++;
        if (k==pattern_size) match[i] = 1;
        D &= ~(1<<s);
    }
}


__device__
static void aoso2_verify_large(
    unsigned char *text, int text_size,
    unsigned char *pattern, int pattern_size,
    int *match, int j, int q, unsigned int D, unsigned int mm, int p_len)
{
    unsigned int s;
    int c, k, i;

    D = (D & mm)^mm;
    while (D != 0) {
        s = LOG2(D);
        c = -(p_len/q-1)*q-s/(p_len/q);
        k = 0; 
        i=j+c;
        if (i>=0 && i<=text_size-pattern_size)
            while (k<p_len && pattern[k]==text[i+k]) k++;
        if (k==pattern_size) match[i] = 1;
        D &= ~(1<<s);
    }
}

__global__
void average_shift_optimal_or_large(
    unsigned char *text, int text_size,
    unsigned char *pattern, int pattern_size,
    unsigned int *B, unsigned int mm,
    int stride_length, int *match)
{
    unsigned int upper_limit;
    int idx = (threadIdx.x + blockIdx.x*blockDim.x) * stride_length;
    if ( idx > text_size - pattern_size ) return;

    if (idx <= text_size - pattern_size - stride_length)
        upper_limit = idx + stride_length + pattern_size;
    else
        upper_limit = text_size;

    int q = 2;
    int p_len = 32;


    /* Searching */
    unsigned int D = ~0;
    int j = idx;
    while (j < upper_limit) {
        D = ((D & ~mm)<<1)|B[text[j]];
        if ((D & mm) != mm)
            aoso2_verify_large(text, text_size, pattern, pattern_size, match,
                                j, q, D, mm, p_len);
        j += q;
    }
}

__global__
void average_shift_optimal_or(
    unsigned char *text, int text_size,
    unsigned char *pattern, int pattern_size,
    unsigned int *B, unsigned int mm,
    int stride_length, int *match)
{
    unsigned int upper_limit;
    int idx = (threadIdx.x + blockIdx.x*blockDim.x) * stride_length;
    if ( idx > text_size - pattern_size ) return;

    if (idx <= text_size - pattern_size - stride_length)
        upper_limit = idx + stride_length + pattern_size;
    else
        upper_limit = text_size;

    int q = 2;

    /* Searching */
    unsigned int D = ~0;
    int j = idx;
    while (j < upper_limit) {
        D = ((D & ~mm)<<1)|B[text[j]];
        if ((D & mm) != mm)
            aoso2_verify(text, text_size, pattern, pattern_size, match,
                          j, q, D, mm);
        j += q;
    }
}
