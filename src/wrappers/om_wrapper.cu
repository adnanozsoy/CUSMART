
#include "om_wrapper.h"

#include "algos/om.cuh"
#include "wrapper_helpers.h"
#include "util/cutil.cuh"
#include "util/tictoc.h"

#include <stdlib.h>

search_info optimal_mismatch_wrapper(search_parameters params){
	
	cuda_time kernel_time = {0}, total_time = {0};
	unsigned char *d_text, *d_pattern;
	int *d_match;
	unsigned int grid_dim, block_dim;
	
	setup_timers(&kernel_time, &total_time);
	get_kernel_configuration(params, &grid_dim, &block_dim);

	int *d_adaptedGs, *d_qsBc;
	ompattern *d_pat;
	
	gpuErrchk( cudaMalloc(&d_adaptedGs, params.pattern_size * sizeof(int)) );
	gpuErrchk( cudaMalloc(&d_qsBc, SIGMA * sizeof(int)) );
	gpuErrchk( cudaMalloc(&d_pat, params.pattern_size * sizeof(ompattern)) );

	// Setup: malloc > timer start > memset/memcpy
	wrapper_setup(params, &d_text, &d_pattern, &d_match);
	

	/* Preprocessing */
	TicTocTimer preprocess_timer_start = tic();
	int qsBc[SIGMA];
	int *adaptedGs = (int *)malloc(params.pattern_size * sizeof(int));
	ompattern *pat = (ompattern *)malloc((params.pattern_size+1) * sizeof(ompattern));
	om_preprocess(params.pattern, (params.pattern_size+1), pat, qsBc, adaptedGs);
	double preprocess_duration = toc(&preprocess_timer_start) * 1000;
	
	gpuErrchk( cudaMemcpy(d_adaptedGs, adaptedGs, params.pattern_size * sizeof(int), cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(d_qsBc, qsBc, SIGMA * sizeof(int), cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(d_pat, pat, params.pattern_size * sizeof(ompattern), cudaMemcpyHostToDevice) );

	//Kernel run
	gpuErrchk( cudaEventRecord(kernel_time.start) );
	optimal_mismatch<<<grid_dim, block_dim>>>(
		d_text, params.text_size, d_pattern, params.pattern_size, 
		d_adaptedGs, d_qsBc, d_pat, params.stride_length, d_match);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaEventRecord(kernel_time.stop) );

	search_info timers = {0};
	timers.preprocess_duration = preprocess_duration;
	// Teardown: copy match back > timer stop > free
	wrapper_teardown(params, &timers, d_text, d_pattern, d_match);
	
	// Release memory
	gpuErrchk( cudaFree(d_adaptedGs) );
	gpuErrchk( cudaFree(d_qsBc) );
	gpuErrchk( cudaFree(d_pat) );

	free(adaptedGs);
	free(pat);

	return timers;
}
