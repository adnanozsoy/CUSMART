#ifndef AC_CUH
#define AC_CUH

__host__
void preKmpAC(unsigned char *pattern, int pattern_size, int shift_array[]);

__global__
void apostolico_crochemore( 
        unsigned char *text, int text_size, 
        unsigned char *pattern, int pattern_size,
        int ell, int *shift_array, int stride_length, int *match);

#endif
