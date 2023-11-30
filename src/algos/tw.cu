#include "tw.cuh"



__global__ void two_way1(unsigned char *text, unsigned long text_size,
	unsigned char *pattern, int pattern_size, int per, int ell, 
	int search_len, int *match)
{
	unsigned long thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned long start_inx = thread_id * search_len;
	
	unsigned long boundary = start_inx + search_len;
	boundary = boundary > text_size ? text_size : boundary;
	int i;
	int memory = -1;
	unsigned long j = start_inx;

	while (j < boundary + pattern_size - 1 && j <= text_size - pattern_size) {
		i = MAX(ell, memory) + 1;
		while (i < pattern_size && pattern[i] == text[i + j])
            ++i;
		
		if (i >= pattern_size) {
            i = ell;
            while (i > memory && pattern[i] == text[i + j])
               --i;
             
            if (i <= memory)
               match[j] = 1;
            j += per;
            memory = pattern_size - per - 1;
         }
         else {
            j += (i - ell);
            memory = -1;
         }
	}
}

__global__ void two_way2(unsigned char *text, unsigned long text_size,
	unsigned char *pattern, int pattern_size, int per, int ell, 
	int search_len, int *match)
{
	unsigned long thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned long start_inx = thread_id * search_len;
	
	unsigned long boundary = start_inx + search_len;
	boundary = boundary > text_size ? text_size : boundary;
	int i;
	unsigned long j = start_inx;
	
	per = MAX(ell + 1, pattern_size - ell - 1) + 1;
	while (j < boundary + pattern_size - 1 && j <= text_size - pattern_size) {
		 i = ell + 1;
         while (i < pattern_size && pattern[i] == text[i + j])
            ++i;
         if (i >= pattern_size) {
            i = ell;
            while (i >= 0 && pattern[i] == text[i + j])
               --i;
            if (i < 0)
               match[j] = 1;
            j += per;
         }
         else
            j += (i - ell);      
	}
}




/* Computing of the maximal suffix for <= */
int maxSuf(unsigned char *x, int m, int *p) {
   int ms, j, k;
   char a, b;

   ms = -1;
   j = 0;
   k = *p = 1;
   while (j + k < m) {
      a = x[j + k];
      b = x[ms + k];
      if (a < b) {
         j += k;
         k = 1;
         *p = j - ms;
      }
      else
         if (a == b)
            if (k != *p)
               ++k;
            else {
               j += *p;
               k = 1;
            }
         else { /* a > b */
            ms = j;
            j = ms + 1;
            k = *p = 1;
         }
   }
   return(ms);
}
 
/* Computing of the maximal suffix for >= */
int maxSufTilde(unsigned char *x, int m, int *p) {
   int ms, j, k;
   char a, b;

   ms = -1;
   j = 0;
   k = *p = 1;
   while (j + k < m) {
      a = x[j + k];
      b = x[ms + k];
      if (a > b) {
         j += k;
         k = 1;
         *p = j - ms;
      }
      else
         if (a == b)
            if (k != *p)
               ++k;
            else {
               j += *p;
               k = 1;
            }
         else { /* a < b */
            ms = j;
            j = ms + 1;
            k = *p = 1;
         }
   }
   return(ms);
}
