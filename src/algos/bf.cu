
#include "bf.cuh"

__global__ void brute_force(unsigned char *text, unsigned long text_size, 
                            unsigned char *pattern, int pattern_size, 
                            int *match){

    unsigned long tid = threadIdx.x + blockIdx.x*blockDim.x;

    if (tid <= text_size - pattern_size){

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

