

#include "bww.cuh"

__global__
void bitparallel_wide_window(
    unsigned char *text, unsigned long text_size,
    unsigned char *pattern, int pattern_size,
    unsigned int *B, unsigned int *C, unsigned int s, unsigned int t,
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
    for (int k = idx + pattern_size - 1; k < idx + stride_length;
         k += pattern_size) {
        /* Left to right scanning */
        int r = 0;
        unsigned int pre = 0;
        unsigned int left = 0;
        unsigned int R = ~0;
        unsigned int cur = s;
        while (R != 0 && k+r < upper_limit) {
            R &= B[text[k+r]];
            ++r;
            if ((R & s) != 0) {
                pre |= cur;
                left = max(left, pattern_size+1-r);
            }
            R <<= 1;
            cur >>= 1;
        }
        /* Right to left scanning */
        unsigned int L = ~0;
        cur = 1;
        int ell = 0;
        while (L != 0 && left > ell) {
            L &= C[text[k-ell]];
            if ((L&t) != 0 && (cur&pre) != 0) match[k-ell] = 1;
            L <<= 1;
            cur <<= 1;
            ++ell;
        }
    }
}

/*
 * Bitparallel Wide Window algorithm designed for large patterns
 * The present implementation searches for prefixes of the pattern of length 32.
 * When an occurrence is found the algorithm tests for the whole occurrence of the pattern
 */

__global__
void bitparallel_wide_window_large(
    unsigned char *text, unsigned long text_size,
    unsigned char *pattern, int pattern_size,
    unsigned int *B, unsigned int *C, unsigned int s, unsigned int t,
    int stride_length, int *match)

{
    unsigned int upper_limit;
    int idx = (threadIdx.x + blockIdx.x*blockDim.x) * stride_length;
    if ( idx > text_size - pattern_size ) return;

    if (idx <= text_size - pattern_size - stride_length)
        upper_limit = idx + stride_length + pattern_size;
    else
        upper_limit = text_size;

    int p_len = 30;

    /* Searching */
    for (int k = idx + p_len - 1; k < idx + stride_length; k += p_len) {
        /* Left to right scanning */
        int r = 0;
        unsigned int pre = 0;
        unsigned int left = 0;
        unsigned int R = ~0;
        unsigned int cur = s;
        while (R != 0 && k+r < upper_limit) {
            R &= B[text[k+r]];
            ++r;
            if ((R & s) != 0) {
                pre |= cur;
                left = max(left, p_len+1-r);
            }
            R <<= 1;
            cur >>= 1;
        }
        /* Right to left scanning */
        unsigned int L = ~0;
        cur = 1;
        int ell = 0;
        while (L != 0 && left > ell) {
            L &= C[text[k-ell]];
            if ((L&t) != 0 && (cur&pre) != 0) match[k-ell] = 1;
            {
                int j = p_len;
                int first = k-ell;
                while (j<pattern_size && pattern[j]==text[first+j]) j++;
                if (j==pattern_size) match[first] = 1;
            }
            L <<= 1;
            cur <<= 1;
            ++ell;
        }
    }
}




