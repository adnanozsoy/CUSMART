#ifndef SVM_CUH
#define SVM_CUH

#include "include/define.cuh"

__global__
void svm(unsigned char *text, int text_size, 
        unsigned char *pattern, int pattern_size,
        unsigned int *cv, int stride_length, int *match);

__global__
void svm_large(unsigned char *text, int text_size, 
        unsigned char *pattern, int pattern_size,
        unsigned int *cv, int stride_length, int *match);

#endif
