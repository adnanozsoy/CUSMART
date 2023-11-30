
#include "dfa.cuh"

__global__ void deterministic_finite_automaton(
	unsigned char *text, int text_size, 
	unsigned char *pattern, int pattern_size,int *ttransSMA, 
	int stride_length, int *match)
{
    int upper_limit;
    int idx = (threadIdx.x + blockIdx.x*blockDim.x) * stride_length;
    if ( idx > text_size - pattern_size ) return;

    if (idx <= text_size - pattern_size - stride_length)
        upper_limit = idx + stride_length + pattern_size;
    else
        upper_limit = text_size;

	int state = 0;
	for (int j = idx; j < upper_limit; j++){		
			//state = getSMA(state, y[i]);
			state = ttransSMA[state*256 + text[j]];
			if (state == pattern_size){
				//printf("Thread id: %d, The pattern was found in position %d \n", thread_id, j - m + 1);
				match[j - pattern_size + 1] = 1;
			}		
	}

}

void preSMA(unsigned char *x, int m, int *ttransSMA) {
   int i, j,state, target, oldTarget;
   int c;

   memset(ttransSMA, 0, SIGMA*sizeof(int));
   for (state = 0, i = 0; i < m; ++i) {
      oldTarget = getSMA(state, x[i]);
      target = state+1;
      setSMA(state, x[i], target);
      for (j=0, c=0; j < SIGMA; ++c, ++j)
         setSMA(target, c, getSMA(oldTarget, c));
      state = target;
   }
}
