#include "bom.cuh"
#include <stdio.h>

__device__
int d_getTransition(unsigned char *x, int p, element_index *L, char c) {
   element_index cell;
   if (p > 0 && x[p - 1] == c) return(p - 1);
   else {
      cell = L[p];
      while (cell.next_index > -1 || cell.element > -1)
         if (x[cell.element] == c)
            return(cell.element);
         else{
			 if(cell.next_index < 0)
				return(UNDEFINED);
			 else
				cell = L[cell.next_index];
		  }
      return(UNDEFINED);
   }
}


__global__ 
void backward_oracle_matching(unsigned char *text, unsigned long text_size,
			   unsigned char *pattern, int pattern_size,
			   element_index *L, char *T,
			   int search_len, int *match) {
	
	unsigned long thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned long start_inx = thread_id * search_len;

	unsigned long boundary = start_inx + search_len + pattern_size - 1;
	boundary = boundary > text_size ? text_size : boundary;
	int i, p, q, shift, period;
	unsigned long j = start_inx;

	while (j < boundary && j <= text_size - pattern_size) {
	    i = pattern_size - 1;
	    p = pattern_size;
		shift = pattern_size;
		
		while (i + j >= start_inx && (q = d_getTransition(pattern, p, L, text[i + j])) != UNDEFINED) {
			p = q;
			if (T[p] == TRUE) {
			   period = shift;
			   shift = i;
			}
			--i;
		}
		
		if (i < 0) {
			match[j] = 1;
			shift = period;
		}
		
		j += shift;
	}
}



int getTransition(unsigned char *x, int p, List L[], char c) {
   List cell;
   if (p > 0 && x[p - 1] == c) return(p - 1);
   else {
      cell = L[p];
      while (cell != NULL)
         if (x[cell->element] == c)
            return(cell->element);
         else
            cell = cell->next;
      return(UNDEFINED);
   }
}



void setTransition(int p, int q, List L[]) {
   List cell;
   cell = (List)malloc(sizeof(struct _cell));
   if (cell == NULL)
      exit(1);//error("BOM/setTransition");
   cell->element = q;
   cell->next = L[p];
   //printf("\n+++++ prior L[P] is %d for p= %d\n", L[p], p);
   L[p] = cell;
   //printf("\n--------index is %d itself adres is %d----ekement is %d\n", p, cell, q);
}



void oracle(unsigned char *x, int m, char T[], List L[]) {
   int i, p, q;
   int S[XSIZE + 1];
   char c;
   S[m] = m + 1;
   for (i = m; i > 0; --i) {
      c = x[i - 1];
      p = S[i];
      while (p <= m &&
             (q = getTransition(x, p, L, c)) ==
             UNDEFINED) {
         setTransition(p, i - 1, L);
         p = S[p];
      }
      S[i - 1] = (p == m + 1 ? m : q);
   }
   p = 0;
   while (p <= m) {
      T[p] = TRUE;
      p = S[p];
   }
}
