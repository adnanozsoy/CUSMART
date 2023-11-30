
#include "svm_wrapper.h"
#include "algos/svm.cuh"
#include "wrapper_helpers.h"
#include "util/cutil.cuh"
#include "util/tictoc.h"
#include <stdio.h>

search_info shift_vector_matching_wrapper(search_parameters params){

	cuda_time kernel_time = {0}, total_time = {0};
	unsigned char *d_text, *d_pattern;
	int *d_match;
	unsigned int grid_dim, block_dim;

	setup_timers(&kernel_time, &total_time);
	get_kernel_configuration(params, &grid_dim, &block_dim);

	unsigned int *cv = (unsigned int*)malloc(SIGMA * sizeof(unsigned int));
	unsigned int *d_cv;
	gpuErrchk( cudaMalloc((void**)&d_cv, SIGMA * sizeof(unsigned int)) );

	// Setup: malloc > timer start > memset/memcpy
	wrapper_setup(params, &d_text, &d_pattern, &d_match);



	/* Preprocessing */
	TicTocTimer preprocess_timer_start = tic();
	unsigned int ONE = 1;
	unsigned int tmp = (~0);
	int psize = params.pattern_size > 32 ? 32 : params.pattern_size;
	tmp >>= (32-psize);
	for(int j = 0; j < SIGMA; j++) cv[j] = tmp;
	tmp = ~ONE;
	for(int j = psize-1; j >= 0; j--) {
	  cv[params.pattern[j]] &= tmp;
	  tmp <<= 1;
	  tmp |= 1;
	}
	double preprocess_duration = toc(&preprocess_timer_start) * 1000;
	
	gpuErrchk( cudaMemcpy(d_cv, cv, SIGMA * sizeof(int), cudaMemcpyHostToDevice) );

	// Kernel run
	gpuErrchk( cudaEventRecord(kernel_time.start) );

	if (params.pattern_size <= 32)
		svm<<<grid_dim, block_dim>>>(
		d_text, params.text_size, d_pattern, params.pattern_size, 
		d_cv, params.stride_length, d_match);
	else
		svm_large<<<grid_dim, block_dim>>>(
		d_text, params.text_size, d_pattern, params.pattern_size, 
		d_cv, params.stride_length, d_match);

	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaEventRecord(kernel_time.stop) );

	search_info timers = {0};
	timers.preprocess_duration = preprocess_duration;
	// Teardown: copy match back > timer stop > free
	wrapper_teardown(params, &timers, d_text, d_pattern, d_match);

	cudaFree(d_cv);
	free(cv);
	return timers;
}
