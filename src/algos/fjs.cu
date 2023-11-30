#include "fjs.cuh"

__global__
void fjs(unsigned char *text, unsigned long text_size, unsigned char *pattern,
	 int pattern_size, int *qsbc, int *kmp, int search_len, int *match){
        int i; 
        unsigned long thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned long start_inx = thread_id * search_len;
	
	unsigned long boundary = start_inx + search_len;
	boundary = boundary > text_size ? text_size : boundary;
	unsigned long s = start_inx;

	while(s<boundary && s<=text_size-pattern_size) {
	      while(s<boundary && s<=text_size-pattern_size && pattern[pattern_size-1]!=text[s+pattern_size-1]) s+=qsbc[text[s+pattern_size]];
	      if (s>boundary && s>text_size-pattern_size) return;
	      i=0; 
	      while(i<pattern_size && pattern[i]==text[s+i]) i++;
	      if (i>=pattern_size) match[s] = 1;
	      s+=(i-kmp[i]);
   }
}

__host__
void preQsBcFJS(unsigned char *pattern, int pattern_size, int qbc[]) {
   int i;
   for (i=0;i<SIGMA;i++)   qbc[i]=pattern_size+1;
   for (i=0;i<pattern_size;i++) qbc[pattern[i]]=pattern_size-i;
}

__host__
void preKmpFJS(unsigned char *pattern, int pattern_size, int kmpNext[]) {
   int i, j;
   i = 0;
   j = kmpNext[0] = -1;
   while (i < pattern_size) {
      while (j > -1 && pattern[i] != pattern[j])
         j = kmpNext[j];
      i++;
      j++;
      if (i<pattern_size && pattern[i] == pattern[j])
         kmpNext[i] = kmpNext[j];
      else
         kmpNext[i] = j;
   }
}
