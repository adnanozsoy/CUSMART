#include "bxs.cuh"

__global__
void bndm_extended_shift(unsigned char *text, int text_size,
			 unsigned char *pattern, int pattern_size,
			 unsigned int *B, int search_len, int *match) {
        unsigned int D; 
	int j, first, k; 
	if (pattern_size<Q) return; 
	//int larger = m>WORD? 1:0; 
	//if (larger) m = WORD; 
	int w = WORD, mq1 = pattern_size-Q+1, nq1 = text_size-Q+1; 
	if (w > pattern_size) w = pattern_size; 
	unsigned int mask = 1<<(w-1); 
       
   	unsigned long thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned long start_inx = thread_id * search_len;
	unsigned long boundary = start_inx + search_len + mq1 - 1;
	boundary = boundary > text_size ? text_size : boundary;
	unsigned long i = start_inx + mq1 - 1;
   
	while (i < boundary && i < nq1) { 
	  D = B[text[i]]; 
	  if ( D ) { 
	    j = i;  
	    first = i-mq1; 
	    do { 
	      j--; 
	      if (D >= mask) { 
		if (j-first) i=j; 
		else { 
		  for (k=pattern_size; text[first+k]==pattern[k-1] && (k); k--); 
		  if ( k==0 ) match[i] = 1; 
		} 
		D = ((D<<1)|1) & B[text[j]]; 
	      } 
	      else D = (D<<1) & B[text[j]]; 
	    } while (D && j>first); 
	  }
	  i+=mq1;
	}
}
