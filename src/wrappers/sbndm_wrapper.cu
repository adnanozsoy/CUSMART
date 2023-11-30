
#include "sbndm_wrapper.h"
#include "algos/sbndm.cuh"
#include "wrapper_helpers.h"
#include "util/cutil.cuh"
#include "util/tictoc.h"

search_info simplified_backward_nondeterministic_dawg_matching_wrapper(search_parameters params){

    cuda_time kernel_time = {0}, total_time = {0};
	unsigned char *d_text, *d_pattern;
	int *d_match;
	unsigned int grid_dim, block_dim;

	setup_timers(&kernel_time, &total_time);
	get_kernel_configuration(params, &grid_dim, &block_dim);
	
	unsigned int *d_B;
	gpuErrchk( cudaMalloc(&d_B, SIGMA * sizeof(unsigned int)) );

	// Setup: malloc > timer start > memset/memcpy
	wrapper_setup(params, &d_text, &d_pattern, &d_match);

	/* Preprocessing */
	TicTocTimer preprocess_timer_start = tic();
	int i, last;
	unsigned int B[SIGMA], s;
	int restore[XSIZE+1], shift;
	
	for (i=0; i<SIGMA; i++)  B[i] = 0;
	for (i=0; i<params.pattern_size; i++) B[params.pattern[params.pattern_size-i-1]] |= (unsigned int)1 << (i + WORD-params.pattern_size);

	last = params.pattern_size;
	s = (unsigned int)(~0) << (WORD-params.pattern_size);
	s = (unsigned int)(~0);
	for (i=params.pattern_size-1; i>=0; i--) {
	  s &= B[params.pattern[i]]; 
	  if (s & ((unsigned int)1<<(WORD-1))) {
		 if (i > 0)  last = i; 
	  }
	  restore[i] = last;
	  s <<= 1;
	}
	shift = restore[0];
	double preprocess_duration = toc(&preprocess_timer_start) * 1000;
	//for (i=0; i<params.pattern_size; i++) params.text[params.text_size+i]=params.pattern[i];		
	
	gpuErrchk( cudaMemcpy(d_B, B, SIGMA * sizeof(unsigned int), cudaMemcpyHostToDevice) );
	

	
	//Kernel run
	gpuErrchk( cudaEventRecord(kernel_time.start) );
	simplified_backward_nondeterministic_dawg_matching<<<grid_dim, block_dim>>>(
				       d_text, params.text_size, d_pattern, params.pattern_size,
				       d_B, shift, params.stride_length, d_match);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaEventRecord(kernel_time.stop) );

	search_info timers = {0};
	timers.preprocess_duration = preprocess_duration;
	// Teardown: copy match back > timer stop > free
	wrapper_teardown(params, &timers, d_text, d_pattern, d_match);
	// Release memory
	gpuErrchk( cudaFree(d_B) );

	return timers;
}
