
#include "fjs_wrapper.h"
#include "algos/fjs.cuh"
#include "wrapper_helpers.h"
#include "util/cutil.cuh"
#include "util/tictoc.h"

#include <stdlib.h>

search_info fjs_wrapper(search_parameters params){
  
        cuda_time kernel_time = {0}, total_time = {0};
	unsigned char *d_text, *d_pattern;
	int *d_match;
	unsigned int grid_dim, block_dim;
	
	setup_timers(&kernel_time, &total_time);
	get_kernel_configuration(params, &grid_dim, &block_dim);
	
	int *d_qsBc, *d_kmp;
	gpuErrchk( cudaMalloc(&d_qsBc, SIGMA * sizeof(int)) );
	gpuErrchk( cudaMalloc(&d_kmp, (params.pattern_size+1) * sizeof(int)) );
	
	// Setup: malloc > timer start > memset/memcpy
	wrapper_setup(params, &d_text, &d_pattern, &d_match);
	
	/* Preprocessing */
	TicTocTimer preprocess_timer_start = tic();
	int *h_kmp, *h_qsBc;
	h_kmp = (int *)malloc((params.pattern_size+1) * sizeof(int));
	h_qsBc = (int *)malloc(SIGMA * sizeof(int));   
	
	preKmpFJS(params.pattern, params.pattern_size, h_kmp);
	preQsBcFJS(params.pattern, params.pattern_size, h_qsBc);
	double preprocess_duration = toc(&preprocess_timer_start) * 1000;
	
	gpuErrchk( cudaMemcpy(d_qsBc, h_qsBc, SIGMA * sizeof(int), cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(d_kmp, h_kmp, (params.pattern_size+1) * sizeof(int), cudaMemcpyHostToDevice) );
	
	// Kernel run
	gpuErrchk( cudaEventRecord(kernel_time.start) );
	fjs<<<grid_dim, block_dim>>>(d_text, params.text_size, d_pattern,
				       params.pattern_size, d_qsBc, d_kmp, params.stride_length, d_match);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaEventRecord(kernel_time.stop) );
	
	search_info timers = {0};
	timers.preprocess_duration = preprocess_duration;
	// Teardown: copy match back > timer stop > free
	wrapper_teardown(params, &timers, d_text, d_pattern, d_match);
	
	// Release memory
	gpuErrchk( cudaFree(d_kmp) );
	gpuErrchk( cudaFree(d_qsBc) ); 
	free(h_kmp);
	free(h_qsBc); 
	
	return timers;
}
