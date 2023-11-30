#include "smoa.cuh"



//this function only return 1 if equal otherwise return 0
__device__ int isEqual3(unsigned char *str_a, unsigned char *str_b, int len){
	int equal = 1;
	unsigned i = 0;
	while ((i < len) && equal){		
		if (str_a[i] != str_b[i]){			
			equal = 0;
		}
		i++;
	}
	return equal;
}


/* Compute the next maximal suffix. */
__device__ void nextMaximalSuffix(unsigned char *x, int m,
                       int *i, int *j, int *k, int *p) {
   char a, b;
 
   while (*j + *k < m) {
      a = x[*i + *k];
      b = x[*j + *k];
      if (a == b)
         if (*k == *p) {
            (*j) += *p;
            *k = 1;
         }
         else
            ++(*k);
      else
         if (a > b) {
            (*j) += *k;
            *k = 1;
            *p = *j - *i;
         }
         else {
            *i = *j;
            ++(*j);
            *k = *p = 1;
         }
   }
}


__global__ void string_matching_ordered_alphabet(unsigned char *text, unsigned long text_size,
	unsigned char *pattern, int pattern_size, int search_len, int *match)
{
	unsigned long thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned long start_inx = thread_id * search_len;
	
	unsigned long boundary = start_inx + search_len;	
	boundary = boundary > text_size ? text_size : boundary;
	unsigned long j = start_inx;
	
	int i, ip, jp, k, p;
	ip = -1;
	i = jp = 0;
	k = p = 1;
	while (j < boundary + pattern_size - 1 && j <= text_size - pattern_size) {
		while (i + j < text_size && i < pattern_size && pattern[i] == text[i + j])
			++i;
		if(i==0){
			++j;
			ip = -1;
			jp = 0;
			k = p = 1;		
		}
		else{
			if(i >= pattern_size)
				match[j] = 1;
			
			nextMaximalSuffix(text + j, i+1, &ip, &jp, &k, &p);
			if (ip < 0 || (ip < p && isEqual3(text + j, text + j + p, ip + 1) == 0)) {
				j += p;
				i -= p;
				if (i < 0)
					i = 0;
				if (jp - ip > p)
					jp -= p;
				else {
					ip = -1;
					jp = 0;
					k = p = 1;
				}			
			}
			else{
				j += (MAX(ip + 1, MIN(i - ip - 1, jp + 1)) + 1);
				i = jp = 0;
				ip = -1;
				k = p = 1;			
			}
		}
	
	}
}

