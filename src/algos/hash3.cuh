#ifndef HASH3_CUH
#define HASH3_CUH

#define WSIZE 256
#define RANK3 3 


__global__
void hash3( 
        unsigned char *text, int text_size, 
        unsigned char *pattern, int pattern_size,
        int *shift, int sh1, int stride_length, int *match);

#endif