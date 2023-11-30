
#include "zt_wrapper.h"
#include "algos/zt.cuh"
#include "wrapper_helpers.h"
#include "util/cutil.cuh"
#include "stddef.h"
#include "util/tictoc.h"

search_info zhu_takaoka_wrapper(search_parameters params){

        cuda_time kernel_time = {0}, total_time = {0};
	unsigned char *d_text, *d_pattern;
	int *d_match;
	unsigned int grid_dim, block_dim;

	setup_timers(&kernel_time, &total_time);
	get_kernel_configuration(params, &grid_dim, &block_dim);
	
	int *d_bmGs, **d_ztBc, i;
	int **d_ztBc2 = (int **)malloc(SIGMA * sizeof(int *));
	gpuErrchk( cudaMalloc(&d_bmGs, (params.pattern_size+1) * sizeof(int)) );
	gpuErrchk( cudaMalloc((void***)&d_ztBc, SIGMA * sizeof(int *)) );

	// Setup: malloc > timer start > memset/memcpy
	wrapper_setup(params, &d_text, &d_pattern, &d_match);

	/* Preprocessing */
	TicTocTimer preprocess_timer_start = tic();
	int *h_bmGs = (int *)malloc((params.pattern_size+1) * sizeof(int));
	int h_ztBc[SIGMA][SIGMA];
	
	preZtBcZT(params.pattern, params.pattern_size, h_ztBc);
	preBmGsZT(params.pattern, params.pattern_size, h_bmGs);
	double preprocess_duration = toc(&preprocess_timer_start) * 1000;
	
	gpuErrchk( cudaMemcpy(d_bmGs, h_bmGs, (params.pattern_size+1) * sizeof(int), cudaMemcpyHostToDevice) );
	for(i = 0; i < SIGMA; i++){
	  gpuErrchk( cudaMalloc((void**) &(d_ztBc2[i]), SIGMA*sizeof(int)) ); 
	  gpuErrchk( cudaMemcpy(d_ztBc2[i], h_ztBc[i], SIGMA*sizeof(int), cudaMemcpyHostToDevice) );
	}
	gpuErrchk( cudaMemcpy(d_ztBc, d_ztBc2, SIGMA*sizeof(float *), cudaMemcpyHostToDevice) );
	
	//Kernel run
	gpuErrchk( cudaEventRecord(kernel_time.start) );
	zhu_takaoka<<<grid_dim, block_dim>>>(
				       d_text, params.text_size, d_pattern, params.pattern_size,
				       d_bmGs, d_ztBc, params.stride_length, d_match);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaEventRecord(kernel_time.stop) );

	search_info timers = {0};
	timers.preprocess_duration = preprocess_duration;
	// Teardown: copy match back > timer stop > free
	wrapper_teardown(params, &timers, d_text, d_pattern, d_match);
	// Release memory
	gpuErrchk( cudaFree(d_bmGs) );
       	gpuErrchk( cudaFree(d_ztBc) );
	for(i = 0; i < SIGMA; i++){
	  gpuErrchk( cudaFree(d_ztBc2[i]) ); 
	}
	free(h_bmGs);
	return timers;
}
