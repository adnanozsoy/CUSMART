#ifndef BFS_CUH
#define BFS_CUH

#include "include/define.cuh"

__host__
void PreBFS(unsigned char *pattern, int pattern_size, int *bm_gs);

__global__
void backward_fast_search(
	unsigned char *text, int text_size,
	unsigned char *pattern, int pattern_size,
    int *bc, int *gs, int stride_length, int *match);

#endif
