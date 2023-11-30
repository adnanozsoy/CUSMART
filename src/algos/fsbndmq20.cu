#include "fsbndmq20.cuh"

#define Q 2
#define F 0

__global__
void forward_simplified_bndm_qgram_schar(
    unsigned char *text, int text_size,
    unsigned char *pattern, int pattern_size,
    unsigned int *B, int mm, int sh, int m1,
    int stride_length, int *match)
{
    if (pattern_size<Q) return;

    int upper_limit;
    int idx = (threadIdx.x + blockIdx.x * blockDim.x) * stride_length;
    if ( idx > text_size - pattern_size ) return;
    if (idx <= text_size - pattern_size - stride_length)
        upper_limit = idx + pattern_size + stride_length;
    else
        upper_limit = text_size;

    int larger = pattern_size+F>WORD ? 1 : 0;
    int plen = larger ?  WORD-F : pattern_size;


    /* Searching */
    //if(!memcmp(pattern,text,plen)) match[0] = 1;
    // int end = text_size-pattern_size+plen;
    int j = idx + plen;
    while (j < upper_limit) {
        unsigned int D = B[text[j]];
        D = (D<<1) & B[text[j-1]];
        if (D != 0) {
            int pos = j;
            while (D = (D<<1) & B[text[j-2]]) --j;
            j += mm;
            if (j == pos) {
                if (larger) {
                    int i=plen;
                    while (i<pattern_size &&
                           pattern[i]==text[j-m1+i]) i++;
                    if (i==pattern_size) match[j] = 1;
                }
                else match[j] = 1;
                ++j;
            }
        }
        else j += sh;
    }
}