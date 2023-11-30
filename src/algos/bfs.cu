
#include "bfs.cuh"

#include <stdlib.h>
#include <string.h>

__global__
void backward_fast_search(
    unsigned char *text, int text_size,
    unsigned char *pattern, int pattern_size,
    int *bc, int *gs, int stride_length, int *match)
{
    int j, k, s;

    int upper_limit;
    int idx = (threadIdx.x + blockIdx.x * blockDim.x) * stride_length;
    if ( idx > text_size - pattern_size ) return;
    if (idx <= text_size - pattern_size - stride_length)
        upper_limit = idx + pattern_size + stride_length;
    else
        upper_limit = text_size;

    /* Searching */
    s = idx + pattern_size - 1;
    int first = gs[SIGMA + pattern[0]];
    while (s < upper_limit) {
        while ((k = bc[text[s]]) != 0) s += k;
        for (j = s - 1, k = pattern_size - 1;
            k > 0 && pattern[k - 1] == text[j]; 
            k--, j--);
        if ( k == 0 ) {
            if (s < text_size) match[j] = 1;
            s += first;
        }
        else s += gs[SIGMA * k + text[j]];
    }
}


__host__
void PreBFS(unsigned char *pattern, int pattern_size, int *bm_gs)
{
    int i, j, c, last, suffix_len;
    int* temp = (int*) malloc((pattern_size+1) * sizeof(int));
    suffix_len = 1;
    last = pattern_size - 1;
    for (i = 0; i <= pattern_size; i++)
        for (j = 0; j < SIGMA; j++) bm_gs[SIGMA * i + j] = pattern_size;
    for (i = 0; i <= pattern_size; i++) temp[i] = -1;
    for (i = pattern_size - 2; i >= 0; i--)
        if (pattern[i] == pattern[last]) {
            temp[last] = i;
            last = i;
        }
    suffix_len++;
    while (suffix_len <= pattern_size) {
        last = pattern_size - 1;
        i = temp[last];
        while (i >= 0) {
            if (i - suffix_len + 1 >= 0) {
                if (pattern[i-suffix_len+1] == pattern[last-suffix_len+1]) {
                    temp[last] = i;
                    last = i;
                }
                if (bm_gs[SIGMA*(pattern_size-suffix_len+1)+pattern[i-suffix_len+1]] > pattern_size-1-i)
                    bm_gs[SIGMA*(pattern_size-suffix_len+1)+pattern[i-suffix_len+1]] = pattern_size-1-i;
            }
            else {
                temp[last] = i;
                last = i;
                for (c = 0; c < SIGMA; c++)
                    if (bm_gs[SIGMA*(pattern_size-suffix_len+1)+c] > pattern_size-1-i)
                        bm_gs[SIGMA*(pattern_size-suffix_len+1)+c] = pattern_size-1-i;
            }
            i = temp[i];
        }
        temp[last] = -1;
        suffix_len++;
    }
    free(temp);
}
