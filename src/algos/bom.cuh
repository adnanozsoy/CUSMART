#ifndef BOM_CUH
#define BOM_CUH

#include "include/define.cuh"

#define XSIZE 50
#define UNDEFINED -1
#define TRUE 1
#define FALSE 0

struct element_index{
    int element; 
    int next_index;
};

struct _cell{
    int element; 
    struct _cell *next;
};
 
typedef struct _cell *List;


__global__
void backward_oracle_matching(unsigned char *text, unsigned long text_size,
			   unsigned char *pattern, int pattern_size,
			   element_index *L, char *T,
			   int search_len, int *match);


int getTransition(unsigned char *x, int p, List L[], unsigned char c);
void setTransition(int p, int q, List L[]);
void oracle(unsigned char *x, int m, char T[], List L[]);

#endif
