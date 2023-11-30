#include "skip.cuh"

__global__ void skip_search(
	unsigned char *text, unsigned long text_size,
    unsigned char *pattern, int pattern_size, 
    List *z, int search_len, int *match)
{
	int h, k;
	List ptr;

	unsigned long thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned long start_inx = thread_id * search_len;

	unsigned long boundary = start_inx + search_len + pattern_size;
	boundary = boundary > text_size ? text_size : boundary;
	unsigned long j = start_inx + pattern_size - 1;
	
	/* Searching */
	while (j < boundary && j < text_size){
		for (ptr = z[text[j]]; ptr != NULL; ptr = ptr->next){
			if ((j-ptr->element) <= text_size - pattern_size) {
				k = 0;
				h = j-ptr->element;
				while(k<pattern_size && pattern[k] == text[h+k]) k++;
				if (k>=pattern_size) match[j] = 1;
			}
		}
		j += pattern_size;
	}
}
