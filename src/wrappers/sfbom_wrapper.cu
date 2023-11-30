#include "sfbom_wrapper.h"
#include "algos/sfbom.cuh"
#include "wrapper_helpers.h"
#include "util/cutil.cuh"
#include "stddef.h"
#include "util/tictoc.h"

search_info sfbom_wrapper(search_parameters params)
{
    cuda_time kernel_time = {0}, total_time = {0};
	unsigned char *d_text, *d_pattern;
	int *d_match;
	unsigned int grid_dim, block_dim;

	setup_timers(&kernel_time, &total_time);
	get_kernel_configuration(params, &grid_dim, &block_dim);
	
	int **d_trans;
	int **d_trans2 = (int **)malloc((params.pattern_size+2) * sizeof(int *));
	gpuErrchk( cudaMalloc((void***)&d_trans, (params.pattern_size+2) * sizeof(int *)) );
	int *d_lambda;
	gpuErrchk( cudaMalloc((void**)&d_lambda, (SIGMA*SIGMA)*sizeof(int)) );
	
	// Setup: malloc > timer start > memset/memcpy
	wrapper_setup(params, &d_text, &d_pattern, &d_match);

	/* Preprocessing */
	TicTocTimer preprocess_timer_start = tic();
	int *S = (int *)malloc((params.pattern_size+1) * sizeof(int));
	int **h_trans = (int **)malloc((params.pattern_size+2) * sizeof(int *));;
	int i, j, p, q, iMinus1, c;
	int LAMBDA[SIGMA*SIGMA];
	
	for (i=0; i<=params.pattern_size+1; i++) h_trans[i] = (int *)malloc (sizeof(int)*(SIGMA)); 
	for (i=0; i<=params.pattern_size+1; i++) for (j=0; j<SIGMA; j++) h_trans[i][j]=UNDEFINED; 
	S[params.pattern_size] = params.pattern_size + 1; 
	for (i = params.pattern_size; i > 0; --i) { 
	  iMinus1 = i - 1; 
	  c = params.pattern[iMinus1]; 
	  h_trans[i][c] = iMinus1; 
	  p = S[i]; 
	  while (p <= params.pattern_size && (q = h_trans[p][c]) ==  UNDEFINED) { 
	    h_trans[p][c] = iMinus1; 
	    p = S[p]; 
	  } 
	  S[iMinus1] = (p == params.pattern_size + 1 ? params.pattern_size : q); 
	} 

	/* Construct the First transition table */ 
	for (i=0; i<SIGMA; i++) { 
	  q = h_trans[params.pattern_size][i]; 
	  for (j=0; j<SIGMA; j++) 
	    if (q>=0) { 
	      if ((p=h_trans[q][j])>=0) FT(i,j) = p; 
	      else FT(i,j)=params.pattern_size+params.pattern_size+1; 
	    } 
	    else FT(i,j) = params.pattern_size+params.pattern_size+1; 
	} 
	q = h_trans[params.pattern_size][params.pattern[params.pattern_size-1]]; 
	for (i=0; i<SIGMA; i++) FT(i,params.pattern[params.pattern_size-1]) = q; 
	for (i=0; i<SIGMA; i++) if (FT(params.pattern[0],i)>params.pattern_size) FT(params.pattern[0],i)-=1; 
	double preprocess_duration = toc(&preprocess_timer_start) * 1000;
	
	cudaMemcpy((d_text + params.text_size), params.pattern,
		   params.pattern_size * sizeof(char), cudaMemcpyHostToDevice);
	
	for(i = 0; i < (params.pattern_size+2); i++){
          gpuErrchk( cudaMalloc((void**) &(d_trans2[i]), (SIGMA)*sizeof(int)) );
          gpuErrchk( cudaMemcpy(d_trans2[i], h_trans[i], (SIGMA)*sizeof(int), cudaMemcpyHostToDevice) );
        }

	gpuErrchk( cudaMemcpy(d_lambda, LAMBDA, (SIGMA*SIGMA)*sizeof(int), cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(d_trans, d_trans2, (params.pattern_size+2)*sizeof(int *), cudaMemcpyHostToDevice) );
	
	//Kernel run
	gpuErrchk( cudaEventRecord(kernel_time.start) );
	sfbom<<<grid_dim, block_dim>>>(
				       d_text, params.text_size, d_pattern, params.pattern_size,
				       d_lambda, d_trans, params.stride_length, d_match);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaEventRecord(kernel_time.stop) );
	
	search_info timers = {0};
	timers.preprocess_duration = preprocess_duration;
	// Teardown: copy match back > timer stop > free
	wrapper_teardown(params, &timers, d_text, d_pattern, d_match);
	// Release memory
	gpuErrchk( cudaFree(d_trans) );
	gpuErrchk( cudaFree(d_lambda) );
	for(i = 0; i < (params.pattern_size+1); i++){
          gpuErrchk( cudaFree(d_trans2[i]) );
        }
	free(S);
	for (i=0; i<params.pattern_size+1; i++) free(h_trans[i]);
	return timers;
}
