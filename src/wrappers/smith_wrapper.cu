
#include "smith_wrapper.h"
#include "algos/smith.cuh"
#include "wrapper_helpers.h"
#include "util/cutil.cuh"
#include "util/tictoc.h"

#include <stdlib.h>

search_info smith_wrapper(search_parameters params){
  
        cuda_time kernel_time = {0}, total_time = {0};
	unsigned char *d_text, *d_pattern;
	int *d_match;
	unsigned int grid_dim, block_dim;
	
	setup_timers(&kernel_time, &total_time);
	get_kernel_configuration(params, &grid_dim, &block_dim);
	
	unsigned char *d_qsBc, *d_bmBc;
	gpuErrchk( cudaMalloc(&d_qsBc, SIGMA * sizeof(unsigned char)) );
	gpuErrchk( cudaMalloc(&d_bmBc, SIGMA * sizeof(unsigned char)) );
	
	// Setup: malloc > timer start > memset/memcpy
	wrapper_setup(params, &d_text, &d_pattern, &d_match);
	
	/* Preprocessing */
	TicTocTimer preprocess_timer_start = tic();
	unsigned char *h_bmBc, *h_qsBc;
	int malloc_size = SIGMA * sizeof(unsigned char);
	h_bmBc = (unsigned char *)malloc(malloc_size);
	h_qsBc = (unsigned char *)malloc(malloc_size);   
	
	preBmBcSMITH(params.pattern, params.pattern_size, h_bmBc);
	preQsBcSMITH(params.pattern, params.pattern_size, h_qsBc);
	double preprocess_duration = toc(&preprocess_timer_start) * 1000;
	
	gpuErrchk( cudaMemcpy(d_qsBc, h_qsBc, SIGMA * sizeof(unsigned char), cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(d_bmBc, h_bmBc, SIGMA * sizeof(unsigned char), cudaMemcpyHostToDevice) );
	
	// Kernel run
	gpuErrchk( cudaEventRecord(kernel_time.start) );
	smith<<<grid_dim, block_dim>>>(d_text, params.text_size, d_pattern,
				       params.pattern_size, d_bmBc, d_qsBc, params.stride_length, d_match);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaEventRecord(kernel_time.stop) );
	
	search_info timers = {0};
	timers.preprocess_duration = preprocess_duration;
	// Teardown: copy match back > timer stop > free
	wrapper_teardown(params, &timers, d_text, d_pattern, d_match);
	
	// Release memory
	gpuErrchk( cudaFree(d_bmBc) );
	gpuErrchk( cudaFree(d_qsBc) ); 
	free(h_bmBc);
	free(h_qsBc); 
	
	return timers;
}
