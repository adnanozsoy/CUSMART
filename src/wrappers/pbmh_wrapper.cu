
#include "pbmh_wrapper.h"
#include "algos/pbmh.cuh"
#include "wrapper_helpers.h"
#include "util/cutil.cuh"
#include "util/tictoc.h"

#include <stdlib.h>

search_info pbmh_wrapper(search_parameters params){
	
	cuda_time kernel_time = {0}, total_time = {0};
	unsigned char *d_text, *d_pattern;
	int *d_match;
	unsigned int grid_dim, block_dim;
	
	setup_timers(&kernel_time, &total_time);
	get_kernel_configuration(params, &grid_dim, &block_dim);

	int *d_hbc, *d_v;
	gpuErrchk( cudaMalloc(&d_hbc, SIGMA * sizeof(int)) );
	gpuErrchk( cudaMalloc(&d_v, (params.pattern_size+1) * sizeof(int)) );

	// Setup: malloc > timer start > memset/memcpy
	wrapper_setup(params, &d_text, &d_pattern, &d_match);
	

	/* Preprocessing */
	TicTocTimer preprocess_timer_start = tic();
	int i, j, tmp, h_hbc[SIGMA], FREQ[SIGMA]; 
	int *h_v = (int*)malloc((params.pattern_size+1) * sizeof(int));

	/* Computing the frequency of characters */
	for(i=0; i<SIGMA; i++)	FREQ[i] = 0;
	for(i=0; i<100; i++) FREQ[params.text[i]]++;
	
	/* Preprocessing */
	for(i=0; i<params.pattern_size; i++) h_v[i]=i;
	for(i=params.pattern_size-1; i>0; i--)
	  for(j=0; j<i; j++)
	    if(FREQ[params.pattern[h_v[j]]]>FREQ[params.pattern[h_v[j+1]]]) {   
	      tmp = h_v[j+1];
	      h_v[j+1] = h_v[j];
	      h_v[j] = tmp;
	    }
	for(i=0;i<SIGMA;i++)   h_hbc[i]=params.pattern_size;
	for(i=0;i<params.pattern_size-1;i++) h_hbc[params.pattern[i]]=params.pattern_size-i-1;
	double preprocess_duration = toc(&preprocess_timer_start) * 1000;
	
	gpuErrchk( cudaMemcpy(d_hbc, h_hbc, SIGMA * sizeof(int), cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(d_v, h_v, (params.pattern_size+1) * sizeof(int), cudaMemcpyHostToDevice) );

	//Kernel run
	gpuErrchk( cudaEventRecord(kernel_time.start) );
	bmh_prob<<<grid_dim, block_dim>>>(
		d_text, params.text_size, d_pattern, params.pattern_size, 
		d_hbc, d_v, params.stride_length, d_match);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaEventRecord(kernel_time.stop) );

	search_info timers = {0};
	timers.preprocess_duration = preprocess_duration;
	// Teardown: copy match back > timer stop > free
	wrapper_teardown(params, &timers, d_text, d_pattern, d_match);
	// Release memory
	gpuErrchk( cudaFree(d_hbc) );
	gpuErrchk( cudaFree(d_v) );
	free(h_v);

	return timers;
}
