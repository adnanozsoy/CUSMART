

#include "ac.cuh"

__host__
void preKmpAC (unsigned char *pattern, int pattern_size, int shift_array[]){
    int i, j;
    i = 0;
    j = shift_array[0] = -1;
    while (i < pattern_size) {
        while (j > -1 && pattern[i] != pattern[j])
            j = shift_array[j];
        i++;
        j++;
        if (i < pattern_size && pattern[i] == pattern[j])
            shift_array[i] = shift_array[j];

        else
            shift_array[i] = j;
    }
}

__global__
void apostolico_crochemore( 
        unsigned char *text, int text_size, 
        unsigned char *pattern, int pattern_size,
        int ell, int *shift_array, int stride_length, int *match)
{
    unsigned long idx = (threadIdx.x + blockIdx.x*blockDim.x) * stride_length;
    if ( idx > text_size - pattern_size ) return;

    unsigned long upper_limit;
    if (idx <= text_size - pattern_size - stride_length)
        upper_limit = stride_length + idx;
    else
        upper_limit = text_size - pattern_size;

    /* Searching */
    int i = ell;
    unsigned long j = idx;
    int k = 0;
    while (j <= upper_limit) {
        while (i < pattern_size && pattern[i] == text[i + j])
            ++i;
        if (i >= pattern_size) {
            while (k < ell && pattern[k] == text[j + k])
                ++k;
            if (k >= ell)
                match[j] = 1;
        }
        j += (i - shift_array[i]);
        if (i == ell)
            k = max(0, k - 1);

        else if (shift_array[i] <= ell) {
            k = max(0, shift_array[i]);
            i = ell;
        }

        else {
            k = ell;
            i = shift_array[i];
        }
    }
}


