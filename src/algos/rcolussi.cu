
#include "rcolussi.cuh"
#include "stdlib.h"
#include "string.h"
 
__global__
void reverse_colussi(unsigned char *text, int text_size, 
               unsigned char *pattern, int pattern_size,
               int *h, int *rcBc, int *rcGs, int stride_length, int *match)
{
   int i;
   unsigned int upper_limit;
   int idx = (threadIdx.x + blockIdx.x*blockDim.x) * stride_length;
   if ( idx > text_size - pattern_size ) return;

   if (idx <= text_size - pattern_size - stride_length)
      upper_limit = stride_length + idx;
   else
      upper_limit = text_size - pattern_size;

   /* Searching */
   int j = idx;
   int s = pattern_size;
   while (j <= upper_limit) {
      while (j <= upper_limit && 
         pattern[pattern_size - 1] != text[j + pattern_size - 1]) {
		  s = rcBc[text[j + pattern_size - 1] * (pattern_size + 1) + s];
		  j += s;
      }
      for (i = 1; i < pattern_size && pattern[h[i]] == text[j + h[i]]; ++i);
      if (i >= pattern_size && j <= upper_limit)
         match[j] = 1;
      s = rcGs[i];
      j += s;
   }
}


__host__
void preRc(unsigned char *pattern, int pattern_size, int h[], int rcBc[], int rcGs[]) 
{
   int a, i, j, k, q, r, s;

   int *hmin = (int*)malloc((pattern_size+1) * sizeof(int));
   int *kmin = (int*)malloc((pattern_size+1) * sizeof(int));
   int *link = (int*)malloc((pattern_size+1) * sizeof(int));
   int *rmin = (int*)malloc((pattern_size+1) * sizeof(int));
   int *locc = (int*)malloc(SIGMA * sizeof(int));
 
   for (a = 0; a < SIGMA; ++a)
      locc[a] = -1;
   link[0] = -1;
   for (i = 0; i < pattern_size - 1; ++i) {
      link[i + 1] = locc[pattern[i]];
      locc[pattern[i]] = i;
   }

   for (a = 0; a < SIGMA; ++a)
      for (s = 1; s <= pattern_size; ++s) {
         i = locc[a];
         j = link[pattern_size - s];
         while (i - j != s && j >= 0)
            if (i - j > s)
               i = link[i + 1];
            else
               j = link[j + 1];
         while (i - j > s)
            i = link[i + 1];
         rcBc[a * (pattern_size+1) + s] = pattern_size - i - 1;
      }
 
   k = 1;
   i = pattern_size - 1;
   while (k <= pattern_size) {
      while (i - k >= 0 && pattern[i - k] == pattern[i])
         --i;
      hmin[k] = i;
      q = k + 1;
      while (hmin[q - k] - (q - k) > i) {
         hmin[q] = hmin[q - k];
         ++q;
      }
      i += (q - k);
      k = q;
      if (i == pattern_size)
         i = pattern_size - 1;
   }
 
   memset(kmin, 0, pattern_size * sizeof(int));
   for (k = pattern_size; k > 0; --k)
      kmin[hmin[k]] = k;
 
   for (i = pattern_size - 1; i >= 0; --i) {
      if (hmin[i + 1] == i)
         r = i + 1;
      rmin[i] = r;
   }
 
   i = 1;
   for (k = 1; k <= pattern_size; ++k)
      if (hmin[k] != pattern_size - 1 && kmin[hmin[k]] == k) {
         h[i] = hmin[k];
         rcGs[i++] = k;
      }
   i = pattern_size-1;
   for (j = pattern_size - 2; j >= 0; --j)
      if (kmin[j] == 0) {
         h[i] = j;
         rcGs[i--] = rmin[j];
      }
   rcGs[pattern_size] = rmin[0];

   free(hmin);
   free(kmin);
   free(link);
   free(rmin);
   free(locc);
}