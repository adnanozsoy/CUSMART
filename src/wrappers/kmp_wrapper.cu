#include "kmp_wrapper.h"
#include "algos/kmp.cuh"
#include "wrapper_helpers.h"
#include "util/cutil.cuh"
#include "util/tictoc.h"

#include <stdio.h>

search_info knuth_morris_pratt_wrapper(search_parameters params){

	cuda_time kernel_time = {0}, total_time = {0};
	unsigned char *d_text, *d_pattern;
	int *d_match;
	unsigned int grid_dim, block_dim;

	setup_timers(&kernel_time, &total_time);
	get_kernel_configuration(params, &grid_dim, &block_dim);

	int *d_shift_array;
	int *shift_array = (int*)malloc((params.pattern_size+1)*sizeof(int));
	gpuErrchk( cudaMalloc((void**)&d_shift_array,
		(params.pattern_size+1) * sizeof(int)) );

	// Setup: malloc > timer start > memset/memcpy	
	wrapper_setup(params, &d_text, &d_pattern, &d_match);

	TicTocTimer preprocess_timer_start = tic();
	pre_knuth_morris_pratt(params.pattern, params.pattern_size, shift_array);
	double preprocess_duration = toc(&preprocess_timer_start) * 1000;
	
	gpuErrchk( cudaMemcpy(d_shift_array, shift_array, 
		(params.pattern_size+1) * sizeof(int), cudaMemcpyHostToDevice) );

	// Kernel run
	gpuErrchk( cudaEventRecord(kernel_time.start) );
	knuth_morris_pratt<<<grid_dim, block_dim>>>(
		d_text, params.text_size, 
		d_pattern, params.pattern_size, 
		d_shift_array, params.stride_length, d_match);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaEventRecord(kernel_time.stop) );

	search_info timers = {0};
	timers.preprocess_duration = preprocess_duration;
	// Teardown: copy match back > timer stop > free
	wrapper_teardown(params, &timers, d_text, d_pattern, d_match);
	// Release memory
	gpuErrchk( cudaFree(d_shift_array) );
	free(shift_array);
	return timers;
}
