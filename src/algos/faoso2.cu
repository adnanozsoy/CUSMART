
#include "faoso2.cuh"

__device__
static void faoso2_verify(
    unsigned char *text, int text_size,
    unsigned char *pattern, int pattern_size,
    int *match, int j, int q, int u, unsigned int D, unsigned int mm)
{
    int s, c, mq, v, z, i, k;

    D = (D & mm)^mm;
    mq = pattern_size/q-1;
    while (D != 0) {
        s = LOG2(D);
        v = mq+u;
        c = -mq*q;
        z = s%v-mq;
        c -= (s/v + z*q);
        i = j+c;
        k = 0;
        if (i>=0 && i<=text_size-pattern_size)
            while (k<pattern_size && pattern[k]==text[i+k]) k++;
        if (k==pattern_size) match[i] = 1;
        D &= ~(1<<s);
    }
}


__device__
static void faoso2_verify_large(
    unsigned char *text, int text_size,
    unsigned char *pattern, int pattern_size,
    int *match, int j, int q, int u, unsigned int D, unsigned int mm, int p_len)
{
    int s, c, mq, v, z, i, k;

    D = (D & mm)^mm;
    mq = pattern_size/q-1;
    while (D != 0) {
        s = LOG2(D);
        v = mq+u;
        c = -mq*q;
        z = s%v-mq;
        c -= (s/v + z*q);
        i = j+c;
        k = 0;
        if (i>=0 && i<=text_size-pattern_size)
            while (k<p_len && pattern[k]==text[i+k]) k++;
        if (k==pattern_size) match[i] = 1;
        D &= ~(1<<s);
    }
}

__global__
void fast_average_shift_optimal_or_large(
    unsigned char *text, int text_size,
    unsigned char *pattern, int pattern_size,
    unsigned int *B, unsigned int masq, unsigned int mm,
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
    int u = 2;
    int p_len = 32-u+1;


    /* Searching */
    unsigned int D = ~mm;
    int uq = u*q;
    int uq1 = (u-1)*q;
    int j = idx;
    while (j < upper_limit) {
        D = (D<<1)|(B[text[j]]&~masq);
        D = (D<<1)|(B[text[j+q]]&~masq);
        if ((D & mm) != mm)
            faoso2_verify_large(text, text_size, pattern, pattern_size, match,
                                j+uq1, q, u, D, mm, p_len);
        D &= ~mm;
        j += uq;
    }
}

__global__
void fast_average_shift_optimal_or(
    unsigned char *text, int text_size,
    unsigned char *pattern, int pattern_size,
    unsigned int *B, unsigned int masq, unsigned int mm,
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
    int u = 2;

    /* Searching */
    unsigned int D = ~mm;
    int uq = u*q;
    int uq1 = (u-1)*q;
    int j = idx;
    while (j < upper_limit) {
        D = (D<<1)|(B[text[j]]&~masq);
        D = (D<<1)|(B[text[j+q]]&~masq);
        if ((D & mm) != mm)
            faoso2_verify(text, text_size, pattern, pattern_size, match,
                          j+uq1, q, u, D, mm);
        D &= ~mm;
        j += uq;
    }
}
