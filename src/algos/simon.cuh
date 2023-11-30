#ifndef SIMON_CUH
#define SIMON_CUH

struct _cell{
  int element;
  struct _cell *next;
};

typedef struct _cell* List;

typedef struct shift_struct{
	int *start;
	int *data;
} shift_struct;

__global__ 
void simon(	unsigned char *text, unsigned long text_size, 
			unsigned char *pattern, int pattern_size, 
			int ell, shift_struct *d_shift, int stride_length, int *match);

__device__
int getTransitionSimon(unsigned char *x, int m, int p, shift_struct *S, char c);

__host__
void setTransitionSimon(int p, int q, List L[]);

__host__
int pre_simon(unsigned char *x, int m, List L[]);

__host__
shift_struct* flatten_list_to_array(List *shift_list, int len);

#endif