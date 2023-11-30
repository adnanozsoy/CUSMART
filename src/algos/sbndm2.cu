#include "sbndm2.cuh"

__global__
void simplified_backward_nondeterministic_dawg_unrolled(
   unsigned char *text, unsigned long text_size,
   unsigned char *pattern, int pattern_size,
   unsigned int *B, int shift, int stride_length, int *match)
{
    int upper_limit;
    int idx = (threadIdx.x + blockIdx.x * blockDim.x) * stride_length;
    if (idx > text_size - pattern_size) return;

    if (idx <= text_size - pattern_size - stride_length)
        upper_limit = idx + pattern_size + stride_length;
    else
        upper_limit = text_size;

    int mMinus1 = pattern_size - 1;
    int m2 = pattern_size - 2;

    /* Searching */
    int j = idx + pattern_size;
    while (j < upper_limit) {
        unsigned int D = (B[text[j]] << 1) & B[text[j - 1]];
        if (D != 0) {
            int pos = j;
            while (D = (D << 1) & B[text[j - 2]]) --j;
            j += m2;
            if (j == pos) {
                match[j] = 1;
                j += shift;
            }
        }
        else
            j += mMinus1;
    }
}

__global__
void simplified_backward_nondeterministic_dawg_unrolled_large(
   unsigned char *text, unsigned long text_size,
   unsigned char *pattern, int pattern_size,
   unsigned int *B, int shift, int stride_length, int *match)
{
    int upper_limit;
    int idx = (threadIdx.x + blockIdx.x * blockDim.x) * stride_length;
    if (idx > text_size - pattern_size) return;

    if (idx <= text_size - pattern_size - stride_length)
        upper_limit = idx + pattern_size + stride_length;
    else
        upper_limit = text_size;

    int p_len = 32;
    int diff = pattern_size - p_len;
    int mMinus1 = p_len - 1;
    int m2 = p_len - 2;

    /* Searching */
    int j = idx + pattern_size;
    while (j + diff < upper_limit) {
        unsigned int D = (B[text[j]] << 1) & B[text[j - 1]];
        if (D != 0) {
            int pos = j;
            while (D = (D << 1) & B[text[j - 2]]) --j;
            j += m2;
            if (j == pos) {
                int i = p_len + 1;
                while (i < pattern_size && pattern[i] == text[j-p_len+1+i]) ++i;
                if (i == pattern_size) match[j - p_len + 1] = 1;
                j += shift;
            }
        }
        else
            j += mMinus1;
    }
}
