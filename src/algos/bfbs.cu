
#include "bfbs.cuh"


__global__ void brute_force_block_shared(
                                    unsigned char *text, unsigned long text_size, 
                                    unsigned char *pattern, int pattern_size, 
                                    int stride_length, int *match){
    __shared__ char s_text[SHARED_MEMORY_SIZE];
    int upper_limit, flag;
    unsigned long copy_limit;
    unsigned long tid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned long idx = tid * stride_length;

    copy_limit = blockDim.x * stride_length + pattern_size;
    int text_left = text_size - (blockIdx.x * blockDim.x * stride_length + pattern_size);
    if (copy_limit > text_left)
        copy_limit = text_left;

    for (unsigned long i = threadIdx.x; i < copy_limit; i += blockDim.x)
        s_text[i] = text[i + blockIdx.x*blockDim.x*stride_length];

    __syncthreads();

    if (idx > text_size - pattern_size) return;

    if (idx <= text_size - pattern_size - stride_length)
        upper_limit = stride_length;
    else
        upper_limit = text_size - pattern_size - idx;

    for (int j = 0; j <= upper_limit; j++)
    {
        int k = threadIdx.x * stride_length + j;            
        flag = 1; 
        for (int i = 0; i < pattern_size; i++)
            if (s_text[k + i] != pattern[i]){
                flag = 0;
                break;
            }

        if (flag == 1)
            match[idx+j] = flag;
    }
}

