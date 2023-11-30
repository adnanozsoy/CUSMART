
#include "lbndm.cuh"

__global__
void long_backward_nondeterministic_dawg(
    unsigned char *text, int text_size, 
    unsigned char *pattern, int pattern_size,
    int *B, int k, int m1, int m2, int rmd, int stride_length, int *match)
{

    int upper_limit;
    int idx = (threadIdx.x + blockIdx.x*blockDim.x) * stride_length;
    if ( idx > text_size - pattern_size ) return;

    if (idx <= text_size - pattern_size - stride_length)
        upper_limit = idx + stride_length;
    else
        upper_limit = text_size - pattern_size;

    unsigned int M = 1 << (32-1);
    int i, j, l;

    /* Searching */
    j = idx;
    while (j <= upper_limit) {
        unsigned int D = B[text[j+m1]];
        int last = (D & M) ? pattern_size-k-rmd : pattern_size-rmd;
        l = m2;
        while (D) {
            D = (D << 1) & B[text[j+l]];
            if (D & M) {
                if (l < k+rmd) {
                    unsigned char *yy = text+j;
                    for (int jj=0; jj<k; jj++) {
                        if (pattern_size+jj > text_size-j) break;
                        i = 0;
                        while(i<pattern_size && yy[i]==pattern[i]) i++;
                        if(i>=pattern_size) match[j] = 1;
                        yy++;
                    }
                    break;
                }
                last = l-(k+rmd)+1;
            }
            l -= k;
        }
        j += last;
    }
}
