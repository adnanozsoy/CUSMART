#include "simon.cuh"
#include <stdlib.h>
#include <stdio.h>

__global__ 
void simon(	unsigned char *text, unsigned long text_size, 
			unsigned char *pattern, int pattern_size, 
			int ell, shift_struct *d_shift, int stride_length, int *match){

	int state, range;
	unsigned long j;
	unsigned long idx = (threadIdx.x + blockIdx.x*blockDim.x) * stride_length;
	if ( idx > text_size - pattern_size ) return;


	if (idx <= text_size - pattern_size - stride_length)
		range = stride_length + pattern_size;
	else
		range = text_size - idx;

   	/* Searching */
	for (state = -1, j = idx; j < idx + range; j++) {
		state = getTransitionSimon(pattern, pattern_size, state, d_shift, text[j]);
		if (state >= pattern_size - 1) {
			match[j - pattern_size + 1] = 1;
			state = ell;
		}
	}
}

__device__
int getTransitionSimon(unsigned char *x, int m, int p, shift_struct *S, char c) {
	int start, data;
 
	if (p < m - 1 && x[p + 1] == c)
		return(p + 1);
	else if (p > -1) {
		start = S->start[p];
		data = S->data[start++];
		while (data != -1){
			if (x[data] == c)
				return(data);
		 	else
				data = S->data[start++];
		}
		return(-1);
	}
	else
		return(-1);
}
 
 __host__
void setTransitionSimon(int p, int q, List L[]) {
   List cell;
 
   cell = (List)malloc(sizeof(struct _cell));
   if (cell == NULL)
	  printf("SIMON/setTransition");
   cell->element = q;
   cell->next = L[p];
   L[p] = cell;
}
 
__host__
int pre_simon(unsigned char *x, int m, List L[]) {
   int i, k, ell;
   List cell;
 
   memset(L, 0, (m - 1)*sizeof(List));
   ell = -1;
   for (i = 1; i < m; ++i) {
	  k = ell;
	  cell = (ell == -1 ? NULL : L[k]);
	  ell = -1;
	  if (x[i] == x[k + 1])
		 ell = k + 1;
	  else
		 setTransitionSimon(i - 1, k + 1, L);
	  while (cell != NULL) {
		 k = cell->element;
		 if (x[i] == x[k])
			ell = k;
		 else
			setTransitionSimon(i - 1, k, L);
		 cell = cell->next;
	  }
   }
   return(ell);
}

shift_struct* flatten_list_to_array(List *shift_list, int len) {
	List cell;
	shift_struct *shift = (shift_struct*)malloc(sizeof(shift_struct));
	shift->data = (int*)malloc(2 * len * sizeof(int));
    memset(shift->data, -1, 2 * len * sizeof(int));
    shift->start = (int*)malloc(len * sizeof(int));
    memset(shift->start, -1, len * sizeof(int));

    /* Convert linked list to array format*/
    int k = 0;
    for (int i = 0; i < len; ++i)
    {
    	cell = shift_list[i];
    	shift->start[i] = k;
    	if (cell != NULL)
    	{
    		shift->data[k++] = cell->element;
    		while((cell = cell->next) != NULL)
    			shift->data[k++] = cell->element;
    		k++;
    	}
    	else 
    	    shift->data[k++] = -1;
    }
    return shift;
}