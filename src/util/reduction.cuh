#ifndef REDUCTION_CUH
#define REDUCTION_CUH

void device_reduce_block_atomic(int *in, int* out, int N);

#endif