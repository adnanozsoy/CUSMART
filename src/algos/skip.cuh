#ifndef SKIP_CUH
#define SKIP_CUH

#include "include/define.cuh"

struct _cell{
  int element;
  struct _cell *next;
};

typedef struct _cell *List;

__global__
void skip_search(unsigned char *text, unsigned long text_size,
        unsigned char *pattern, int pattern_size, List z[], int search_len, int *match);

#endif
