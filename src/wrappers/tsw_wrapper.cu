
#include "tsw_wrapper.h"
#include "algos/tsw.cuh"
#include "wrapper_helpers.h"
#include "util/cutil.cuh"
#include "stddef.h"
#include "util/tictoc.h"

search_info two_sliding_window_wrapper(search_parameters params){

        cuda_time kernel_time = {0}, total_time = {0};
	unsigned char *d_text, *d_pattern;
	int *d_match;
	unsigned int grid_dim, block_dim;

	setup_timers(&kernel_time, &total_time);
	get_kernel_configuration(params, &grid_dim, &block_dim);
	
	int **d_brBc_left, **d_brBc_right, i, j;
	int **d_brBc_left2 = (int **)malloc(SIGMA * sizeof(int *));
	int **d_brBc_right2 = (int **)malloc(SIGMA * sizeof(int *));
	gpuErrchk( cudaMalloc((void***)&d_brBc_right, SIGMA * sizeof(int *)) );
	gpuErrchk( cudaMalloc((void***)&d_brBc_left, SIGMA * sizeof(int *)) );
	
	// Setup: malloc > timer start > memset/memcpy
	wrapper_setup(params, &d_text, &d_pattern, &d_match);

	/* Preprocessing */
	TicTocTimer preprocess_timer_start = tic();
	int h_brBc_right[SIGMA][SIGMA]; int h_brBc_left[SIGMA][SIGMA];
	unsigned char *h_x1 = (unsigned char *)malloc((params.pattern_size+1) * sizeof(char));
	
	for (i=params.pattern_size-1, j=0; i>=0; i--, j++) h_x1[j]=params.pattern[i];
	preBrBcTSW(params.pattern, params.pattern_size, h_brBc_left);
	preBrBcTSW(h_x1, params.pattern_size, h_brBc_right);
	double preprocess_duration = toc(&preprocess_timer_start) * 1000;
	
	for(i = 0; i < SIGMA; i++){
	  gpuErrchk( cudaMalloc((void**) &(d_brBc_left2[i]), SIGMA*sizeof(int)) );
          gpuErrchk( cudaMemcpy(d_brBc_left2[i], h_brBc_left[i], SIGMA*sizeof(int), cudaMemcpyHostToDevice) );
	  gpuErrchk( cudaMalloc((void**) &(d_brBc_right2[i]), SIGMA*sizeof(int)) ); 
	  gpuErrchk( cudaMemcpy(d_brBc_right2[i], h_brBc_right[i], SIGMA*sizeof(int), cudaMemcpyHostToDevice) );
	}
	gpuErrchk( cudaMemcpy(d_brBc_left, d_brBc_left2, SIGMA*sizeof(float *), cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(d_brBc_right, d_brBc_right2, SIGMA*sizeof(float *), cudaMemcpyHostToDevice) );
	
	//Kernel run
	gpuErrchk( cudaEventRecord(kernel_time.start) );
	two_sliding_window<<<grid_dim, block_dim>>>(
				       d_text, params.text_size, d_pattern, params.pattern_size,
				       d_brBc_left, d_brBc_right, params.stride_length, d_match);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaEventRecord(kernel_time.stop) );

	search_info timers = {0};
	timers.preprocess_duration = preprocess_duration;
	// Teardown: copy match back > timer stop > free
	wrapper_teardown(params, &timers, d_text, d_pattern, d_match);
	// Release memory
	gpuErrchk( cudaFree(d_brBc_left) );
       	gpuErrchk( cudaFree(d_brBc_right) );
	for(i = 0; i < SIGMA; i++){
	  gpuErrchk( cudaFree(d_brBc_left2[i]) );
	  gpuErrchk( cudaFree(d_brBc_right2[i]) );
	}
	free(h_x1);
	return timers;
}
