
#include "dfah.cuh"

__global__ void high_deterministic_finite_automaton(unsigned char *text, int text_size, 
		unsigned char *pattern, int pattern_size, int *ttransSMA, int *match){

	int check_point = blockDim.x * blockIdx.x + threadIdx.x;
	int i = check_point;
	int state = 0;
	if (i < text_size - pattern_size){
		//state = getSMA(state, y[i]);
		state = ttransSMA[(state)*256+(text[i])];
		while(i < text_size && state > 0 && state < pattern_size){
			i++;
			//state = getSMA(state, y[i]);
			state = ttransSMA[(state)*256+(text[i])];
		}
		if (state==pattern_size){
			//printf("The pattern was found in position %d \n", check_point);
			match[check_point] = 1;
		}
	}
}

void preHSMA(unsigned char *x, int m, int *ttransSMA) {
   int i, j,state, target, oldTarget;
   char c;

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
