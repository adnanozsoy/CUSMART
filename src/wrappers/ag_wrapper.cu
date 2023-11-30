
#include "ag_wrapper.h"
#include "algos/ag.cuh"
#include "wrapper_helpers.h"
#include "util/cutil.cuh"
#include "util/tictoc.h"

#include <stdio.h>

search_info apostolico_giancarlo_wrapper(search_parameters params){
	
	cuda_time kernel_time = {0}, total_time = {0};
	unsigned char *d_text, *d_pattern;
	int *d_match;
	unsigned int grid_dim, block_dim;
	
	setup_timers(&kernel_time, &total_time);
	get_kernel_configuration(params, &grid_dim, &block_dim);

	int *d_bmBc, *d_bmGs, *d_suff;
	int *h_bmGs = (int*)malloc((params.pattern_size) * sizeof(int));
	int *h_suff = (int*)malloc((params.pattern_size) * sizeof(int));
	int *h_bmBc = (int*)malloc(SIGMA * sizeof(int));

	gpuErrchk( cudaMalloc((void**)&d_bmGs, (params.pattern_size) * sizeof(int)) );
	gpuErrchk( cudaMalloc((void**)&d_suff, (params.pattern_size) * sizeof(int)) );
	gpuErrchk( cudaMalloc((void**)&d_bmBc, SIGMA * sizeof(int)) );

	// Setup: malloc > timer start > memset/memcpy
	wrapper_setup(params, &d_text, &d_pattern, &d_match);	

	/* Preprocessing */
	TicTocTimer preprocess_timer_start = tic();
	suffixesAG(params.pattern, params.pattern_size, h_suff);
	preBmGsAG(params.pattern, params.pattern_size, h_bmGs, h_suff);
	preBmBcAG(params.pattern, params.pattern_size, h_bmBc);
	double preprocess_duration = toc(&preprocess_timer_start) * 1000;
	
	gpuErrchk( cudaMemcpy(d_bmGs, h_bmGs, 
		(params.pattern_size) * sizeof(int), cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(d_suff, h_suff, 
		(params.pattern_size) * sizeof(int), cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(d_bmBc, h_bmBc, 
		SIGMA * sizeof(int), cudaMemcpyHostToDevice) );

	//Kernel run
	gpuErrchk( cudaEventRecord(kernel_time.start) );
	ag<<<grid_dim, block_dim>>>(
		d_text, params.text_size, d_pattern, params.pattern_size, 
		d_bmGs, d_bmBc, d_suff, params.stride_length, d_match);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaEventRecord(kernel_time.stop) );

	search_info timers = {0};
	timers.preprocess_duration = preprocess_duration;
	// Teardown: copy match back > timer stop > free
	wrapper_teardown(params, &timers, d_text, d_pattern, d_match);

	// Release memory
	gpuErrchk( cudaFree(d_bmGs) );
	gpuErrchk( cudaFree(d_suff) );
	gpuErrchk( cudaFree(d_bmBc) );
	free(h_bmBc);
	free(h_bmGs);
	free(h_suff);

	return timers;
}
