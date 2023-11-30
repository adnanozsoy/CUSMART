
#include "dfdm_wrapper.h"
#include "algos/dfdm.cuh"
#include "wrapper_helpers.h"
#include "util/cutil.cuh"
#include "util/tictoc.h"

#include <stdlib.h>
#include <string.h>

search_info double_forward_dawg_wrapper(search_parameters params){
	
	cuda_time kernel_time = {0}, total_time = {0};
	unsigned char *d_text, *d_pattern;
	int *d_match;
	unsigned int grid_dim, block_dim;
	
	setup_timers(&kernel_time, &total_time);
	get_kernel_configuration(params, &grid_dim, &block_dim);

	int *d_ttrans, *d_tlength, *d_tsuffix;
	gpuErrchk( cudaMalloc((void**)&d_ttrans, 3 * SIGMA * params.pattern_size * sizeof(int)) );
	gpuErrchk( cudaMalloc((void**)&d_tlength, 3 * params.pattern_size * sizeof(int)) );
	gpuErrchk( cudaMalloc((void**)&d_tsuffix, 3 * params.pattern_size * sizeof(int)) );

	/* Preprocessing */
	TicTocTimer preprocess_timer_start = tic();
	int *h_ttrans = (int *)malloc(3*params.pattern_size*SIGMA*sizeof(int));
	memset(h_ttrans, -1, 3*params.pattern_size*SIGMA*sizeof(int));
	int *h_tlength = (int *)calloc(3*params.pattern_size, sizeof(int));
	int *h_tsuffix = (int *)calloc(3*params.pattern_size, sizeof(int));
	unsigned char *h_tterminal = (unsigned char *)calloc(3*params.pattern_size, sizeof(char));

	int logM = 0;
	int temp = params.pattern_size;
	int a = 2;
	while (temp > a) {
	  ++logM;
	  temp /= a;
	}
	++logM;
	
	int beta = params.pattern_size-1-max(1,min(params.pattern_size/5, 3*logM));
	int alpha = min(params.pattern_size/2,beta-1);
	double preprocess_duration = toc(&preprocess_timer_start) * 1000;
	
	// Setup: malloc > timer start > memset/memcpy
	wrapper_setup(params, &d_text, &d_pattern, &d_match);	

	/* Preprocessing */
	buildSimpleSuffixAutomatonFDM(params.pattern, params.pattern_size, h_ttrans, h_tlength, h_tsuffix, h_tterminal);

	gpuErrchk( cudaMemcpy(d_ttrans, h_ttrans, 
		3 * SIGMA * params.pattern_size * sizeof(int), cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(d_tlength, h_tlength, 
		3 * params.pattern_size * sizeof(int), cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(d_tsuffix, h_tsuffix, 
		3 * params.pattern_size * sizeof(int), cudaMemcpyHostToDevice) );

	//Kernel run
	gpuErrchk( cudaEventRecord(kernel_time.start) );
	double_forward_dawg<<<grid_dim, block_dim>>>(
		d_text, params.text_size, d_pattern, params.pattern_size, 
		d_ttrans, d_tlength, d_tsuffix, alpha, beta, logM, 
		params.stride_length, d_match);

	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaEventRecord(kernel_time.stop) );

	search_info timers = {0};
	timers.preprocess_duration = preprocess_duration;
	// Teardown: copy match back > timer stop > free
	wrapper_teardown(params, &timers, d_text, d_pattern, d_match);

	// Release memory
	gpuErrchk( cudaFree(d_ttrans) );
	gpuErrchk( cudaFree(d_tlength) );
	gpuErrchk( cudaFree(d_tsuffix) );
	free(h_ttrans);
	free(h_tlength);
	free(h_tsuffix);
	free(h_tterminal);

	return timers;
}
