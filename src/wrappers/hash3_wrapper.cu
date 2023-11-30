
#include "hash3_wrapper.h"
#include "algos/hash3.cuh"
#include "wrapper_helpers.h"
#include "util/cutil.cuh"
#include "util/tictoc.h"

#include <stdlib.h>

search_info hash3_wrapper(search_parameters params){

        cuda_time kernel_time = {0}, total_time = {0};
	unsigned char *d_text, *d_pattern;
  	int *d_match;
	unsigned int grid_dim, block_dim;

	setup_timers(&kernel_time, &total_time);
	get_kernel_configuration(params, &grid_dim, &block_dim);

	int *d_shift;
	gpuErrchk( cudaMalloc((void**)&d_shift, WSIZE) );  
	
        // Setup: malloc > timer start > memset/memcpy
	wrapper_setup(params, &d_text, &d_pattern, &d_match);   
	
	/* Preprocessing */
	TicTocTimer preprocess_timer_start = tic();
	int i, sh1, mMinus1, mMinus2, h_shift[WSIZE]; 
	unsigned char h;  
	mMinus1 = params.pattern_size-1; 
	mMinus2 = params.pattern_size-2; 
	for (i = 0; i < WSIZE; ++i) 
	  h_shift[i] = mMinus2; 
	
	h = params.pattern[0]; 
	h = ((h<<1) + params.pattern[1]); 
	h = ((h<<1) + params.pattern[2]); 
	h_shift[h] = params.pattern_size-RANK3; 
	for (i=RANK3; i < mMinus1; ++i) { 
	  h = params.pattern[i-2]; 
	  h = ((h<<1) + params.pattern[i-1]); 
	  h = ((h<<1) + params.pattern[i]); 
	  h_shift[h] = mMinus1-i; 
	} 
	h = params.pattern[i-2]; 
	h = ((h<<1) + params.pattern[i-1]); 
	h = ((h<<1) + params.pattern[i]); 
	sh1 = h_shift[h]; 
	h_shift[h] = 0; 
	if (sh1==0) sh1=1; 
	
	double preprocess_duration = toc(&preprocess_timer_start) * 1000;

	cudaMemcpy((d_text + params.text_size), params.pattern,
		   params.pattern_size * sizeof(unsigned char), cudaMemcpyHostToDevice);
	
	gpuErrchk( cudaMemcpy(d_shift, h_shift, WSIZE, cudaMemcpyHostToDevice) );

	// Kernel run
	gpuErrchk( cudaEventRecord(kernel_time.start) );
	hash3<<<grid_dim, block_dim>>>(d_text, params.text_size, d_pattern,
				       params.pattern_size, d_shift, sh1, params.stride_length, d_match);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaEventRecord(kernel_time.stop) );

	search_info timers = {0};
	timers.preprocess_duration = preprocess_duration;
	// Teardown: copy match back > timer stop > free
	wrapper_teardown(params, &timers, d_text, d_pattern, d_match);

	// Release memory
	gpuErrchk( cudaFree(d_shift) );

	return timers;
}
