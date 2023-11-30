#ifndef AOSO2_CUH
#define AOSO2_CUH

#include "include/define.cuh"

__global__
void average_shift_optimal_or(
    unsigned char *text, int text_size,
    unsigned char *pattern, int pattern_size,
    unsigned int *B, unsigned int mm, 
    int stride_length, int *match);

__global__
void average_shift_optimal_or_large(
    unsigned char *text, int text_size,
    unsigned char *pattern, int pattern_size,
    unsigned int *B, unsigned int mm, 
    int stride_length, int *match);


// LOG2
#define MSK1616         0xFFFF0000U
#define MSK0824         0xFF000000U
#define MSK0808             0xFF00U
#define FIRSTBIT(x) ((x) & MSK1616 ? ((x) & MSK0824 ? \
                     leftbit[((x)>>24) & 0xFF] : 8+leftbit[(x)>>16]) \
                    : ((x) & MSK0808 ? 16+leftbit[(x)>>8] : 24+leftbit[x]))
#define LOG2(x) (31-FIRSTBIT(x))
/* array giving position (1..7) of high-order 1-bit in byte: */
__device__
static int leftbit[] =   {8,7,6,6,5,5,5,5,4,4,4,4,4,4,4,4,
                          3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
                          2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
                          2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
                          1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                          1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                          1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                          1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                          0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                          0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                          0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                          0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                          0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                          0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                          0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                          0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

#endif
