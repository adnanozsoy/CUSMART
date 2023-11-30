
#include "bndmq2_wrapper.h"
#include "algos/bndmq2.cuh"
#include "wrapper_helpers.h"
#include "util/cutil.cuh"
#include "util/tictoc.h"

#include <stdlib.h>

search_info backward_nondeterministic_dawg_qgram_wrapper(search_parameters params){

	cuda_time kernel_time = {0}, total_time = {0};
	unsigned char *d_text, *d_pattern;
	int *d_match;
	unsigned int grid_dim, block_dim;

	setup_timers(&kernel_time, &total_time);
	get_kernel_configuration(params, &grid_dim, &block_dim);

	unsigned int* h_B = (unsigned int*) calloc(SIGMA, sizeof(unsigned int));
	unsigned int* d_B;
	cudaMalloc((void**)&d_B, SIGMA * sizeof(unsigned int));

	// Setup: malloc > timer start > memset/memcpy
	wrapper_setup(params, &d_text, &d_pattern, &d_match);

	/* Preprocessing */
	TicTocTimer preprocess_timer_start = tic();
	int p_len = params.pattern_size > 32 ? 32 : params.pattern_size;
	int s = 1;
	for (int i = p_len-1; i>=0; i--){
		h_B[params.pattern[i]] |= s;
		s <<= 1;
	}
	unsigned int M = 1 << (p_len-1);
	double preprocess_duration = toc(&preprocess_timer_start) * 1000;
	
	cudaMemcpy((d_text + params.text_size), params.pattern, 
		params.pattern_size * sizeof(char), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, SIGMA * sizeof(unsigned int), cudaMemcpyHostToDevice);

	// Kernel run
	gpuErrchk( cudaEventRecord(kernel_time.start) );
	if (params.pattern_size <=32) {
		backward_nondeterministic_dawg_qgram<<<grid_dim, block_dim>>>(
			d_text, params.text_size, d_pattern, params.pattern_size, 
			d_B, M, params.stride_length, d_match);
	}
	else {
		backward_nondeterministic_dawg_qgram_large<<<grid_dim, block_dim>>>(
			d_text, params.text_size, d_pattern, params.pattern_size, 
			d_B, M, params.stride_length, d_match);
	}
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaEventRecord(kernel_time.stop) );

	search_info timers = {0};
	timers.preprocess_duration = preprocess_duration;
	// Teardown: copy match back > timer stop > free
	wrapper_teardown(params, &timers, d_text, d_pattern, d_match);
	// Free memory
	cudaFree(d_B);
	free(h_B);
	return timers;
}
