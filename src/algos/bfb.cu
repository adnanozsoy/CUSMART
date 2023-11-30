
#include "bfb.cuh"

__global__ void brute_force_block(  unsigned char *text, unsigned long text_size, 
                                    unsigned char *pattern, int pattern_size, 
                                    int stride_length, int *match){
    unsigned long tid;
    int range;

    unsigned long idx = (threadIdx.x + blockIdx.x*blockDim.x) * stride_length;
    if ( idx > text_size - pattern_size ) return;

    if (idx <= text_size - pattern_size - stride_length)
        range = stride_length;
    else
        range = text_size - pattern_size - idx;
    
    for (int j = 0; j <= range; ++j)
    {
        tid = idx + j;            
        int flag = 1; 
        for (int i = 0; i < pattern_size; i++){
            if (text[tid+i] != pattern[i]){
                flag = 0;
                break;
            }
        }
        if (flag == 1)
            match[tid] = flag;
    }
}
