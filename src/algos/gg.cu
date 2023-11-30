
#include "gg.cuh"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>


__global__
void galil_giancarlo(  unsigned char *text, int text_size, 
               unsigned char *pattern, int pattern_size,
               int nd, int ell, int *h, int *next, int *shift, 
               int stride_length, int *match)
{
    int i, j, k, last;
    char heavy;

   unsigned int upper_limit;
   int idx = (threadIdx.x + blockIdx.x*blockDim.x) * stride_length;
   if ( idx > text_size - pattern_size ) return;

   if (idx <= text_size - pattern_size - stride_length)
      upper_limit = stride_length + idx;
   else
      upper_limit = text_size - pattern_size;

    if (ell == pattern_size - 1) {
        ell = 0;
        /* Searching for a power of a single character */
        for (j = idx; j < upper_limit; ++j)
            if (pattern[0] == text[j]) {
                ++ell;
                if (ell >= pattern_size)
                    match[j - pattern_size + 1] = 1;
            }
            else
                ell = 0;
    } else {

        /* Searching */
        i = heavy = 0;
        j = idx;
        last = j - 1;
        while (j <= upper_limit) {
            if (heavy && i == 0) {
                k = last - j + 1;
                while (pattern[0] == text[j + k]) ++k;
                if (k <= ell || pattern[ell + 1] != text[j + k]) {
                    i = 0;
                    j += (k + 1);
                    last = j - 1;
                } else {
                    i = 1;
                    last = j + k;
                    j = last - (ell + 1);
                }
                heavy = 0;
            } else {
                while (i < pattern_size && 
                    last < j + h[i] && 
                    pattern[h[i]] == text[j + h[i]]) ++i;
                if (i >= pattern_size || last >= j + h[i]) {
                    match[j] = 1;
                    i = pattern_size;
                }
                if (i > nd)
                    last = j + pattern_size - 1;
                j += shift[i];
                i = next[i];
            }
            heavy = j > last ? 0 : 1;
        }
    }
}

__host__
int preColussiGG(unsigned char *pattern, int pattern_size,  int *h, int *next, int *shift) 
{
   int i, k, nd, q, r, s;
   int *hmax = (int*)malloc((pattern_size+1) * sizeof(int));
   int *kmin = (int*)malloc((pattern_size+1) * sizeof(int));
   int *nhd0 = (int*)malloc((pattern_size+1) * sizeof(int));
   int *rmin = (int*)malloc((pattern_size+1) * sizeof(int));
   
   /* Computation of hmax */
   i = k = 1;
   do {
      while (pattern[i] == pattern[i - k])
         i++;
      hmax[k] = i;
      q = k + 1;
      while (hmax[q - k] + k < i) {
         hmax[q] = hmax[q - k] + k;
         q++;
      }
      k = q;
      if (k == i + 1)
         i = k;
   } while (k <= pattern_size);
   
   /* Computation of kmin */
   memset(kmin, 0, pattern_size*sizeof(int));
   for (i = pattern_size; i >= 1; --i)
      if (hmax[i] < pattern_size)
         kmin[hmax[i]] = i;
   
   /* Computation of rmin */
   for (i = pattern_size - 1; i >= 0; --i) {
      if (hmax[i + 1] == pattern_size)
         r = i + 1;
      if (kmin[i] == 0)
         rmin[i] = r;
      else
         rmin[i] = 0;
   }
   
   /* Computation of h */
   s = -1;
   r = pattern_size;
   for (i = 0; i < pattern_size; ++i)
      if (kmin[i] == 0)
         h[--r] = i;
      else
         h[++s] = i;
   nd = s;
   
   /* Computation of shift */
   for (i = 0; i <= nd; ++i)
      shift[i] = kmin[h[i]];
   for (i = nd + 1; i < pattern_size; ++i)
      shift[i] = rmin[h[i]];
   shift[pattern_size] = rmin[0];
   
   /* Computation of nhd0 */
   s = 0;
   for (i = 0; i < pattern_size; ++i) {
      nhd0[i] = s;
      if (kmin[i] > 0)
         ++s;
   }
   
   
   /* Computation of next */
   for (i = 0; i <= nd; ++i)
      next[i] = nhd0[h[i] - kmin[h[i]]];
   for (i = nd + 1; i < pattern_size; ++i)
      next[i] = nhd0[pattern_size - rmin[h[i]]];
   next[pattern_size] = nhd0[pattern_size - rmin[h[pattern_size - 1]]];

   free(hmax);
   free(kmin);
   free(nhd0);
   free(rmin);
   return(nd);
}
