#include "zt.cuh"

__global__ void zhu_takaoka(unsigned char *text, unsigned long text_size,
			   unsigned char *pattern, int pattern_size, int *bmGs,
			   int **ztBc, int search_len, int *match) {
	
	unsigned long thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned long start_inx = thread_id * search_len;

	unsigned long boundary = start_inx + search_len;
	boundary = boundary > text_size ? text_size : boundary;
	int i;
	unsigned long j = start_inx;

	while (j < boundary && j <= text_size - pattern_size) {
	        i = pattern_size - 1;
		while (i >= 0 && pattern[i] == text[i + j]) {
		  --i;
		}
		if (i < 0) {
		      	match[j] = 1;
			j += bmGs[0];
		}
		else{
		        int a = bmGs[i];
			int b = ztBc[text[j + pattern_size - 2]][text[j + pattern_size - 1]];
			j += ((a) > (b) ? (a) : (b));
		}
	}
}

void suffixesZT(unsigned char *x, int m, int *suff) {
  int f, g, i;

  suff[m - 1] = m;
  g = m - 1;
  for (i = m - 2; i >= 0; --i) {
    if (i > g && suff[i + m - 1 - f] < i - g)
      suff[i] = suff[i + m - 1 - f];
    else {
      if (i < g)
	g = i;
      f = i;
      while (g >= 0 && x[g] == x[g + m - 1 - f])
	--g;
      suff[i] = f - g;
    }
  }
}

void preBmGsZT(unsigned char *x, int m, int bmGs[]) {
  int i, j;
  int *suff = (int *)malloc((m+1) * sizeof(int));

  suffixesZT(x, m, suff);

  for (i = 0; i < m; ++i)
    bmGs[i] = m;
  j = 0;
  for (i = m - 1; i >= 0; --i)
    if (suff[i] == i + 1)
      for (; j < m - 1 - i; ++j)
	if (bmGs[j] == m)
	  bmGs[j] = m - 1 - i;
  for (i = 0; i <= m - 2; ++i)
    bmGs[m - 1 - suff[i]] = m - 1 - i;
  free(suff);
}

void preZtBcZT(unsigned char *x, int m, int ztBc[SIGMA][SIGMA]) {
  int i, j;

  for (i = 0; i < SIGMA; ++i)
    for (j = 0; j < SIGMA; ++j)
      ztBc[i][j] = m;
  for (i = 0; i < SIGMA; ++i)
    ztBc[i][x[0]] = m - 1;
  for (i = 1; i < m - 1; ++i)
    ztBc[x[i - 1]][x[i]] = m - 1 - i;
}
