#include "fs.cuh"

__global__ void fast_search(unsigned char *text, unsigned long text_size,
	unsigned char *pattern, int pattern_size,int *bc, int *gs, 
		int search_len, int *match){
		
	unsigned long thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned long start_inx = thread_id * search_len;
	
	unsigned long boundary = start_inx + search_len;
	boundary = boundary > text_size ? text_size : boundary;
	unsigned long s;
	int k, j;
	
	if (start_inx > pattern_size - 1)
		s = start_inx;
	else
		s = start_inx + pattern_size - 1;	
	
	
	while (s < boundary && s < text_size) {
		while(s < boundary && (k=bc[text[s]]))
			s += k;
		
		if(s < boundary){
			j= 2;		
			while (j <= pattern_size && pattern[pattern_size - j] == text[s-j+1]) 
				j++;
				
			if ( j > pattern_size)
				match[s - j + 2] = 1;
				
			s += gs[pattern_size - j + 1];
		}
	}
}



void Pre_GS(unsigned char *x, int m, int bm_gs[]) {
   int i, j, p, *f;
   f = (int*)malloc((m + 1) * sizeof(int));
   for (i=0; i < m + 1; i++) bm_gs[i]=0;
   f[m]=j=m+1;
   for (i=m; i > 0; i--) {
      while (j <= m && x[i-1] != x[j-1]) {
         if (bm_gs[j] == 0) bm_gs[j]=j-i;
         j=f[j];
      }
      f[i-1]=--j;
   }
   p=f[0];
   for (j=0; j <= m; ++j) {
      if (bm_gs[j] == 0) bm_gs[j]=p;
      if (j == p) p=f[p];
   }
   free(f);
}
