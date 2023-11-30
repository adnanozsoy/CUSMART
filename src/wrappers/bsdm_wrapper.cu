
#include "bsdm_wrapper.h"
#include "algos/bsdm.cuh"
#include "wrapper_helpers.h"
#include "util/cutil.cuh"
#include "util/tictoc.h"

search_info backward_snr_dawg_matching_wrapper(search_parameters params) {

    cuda_time kernel_time = {0}, total_time = {0};
    unsigned char *d_text, *d_pattern;
    int *d_match;
    unsigned int grid_dim, block_dim;

    setup_timers(&kernel_time, &total_time);
    get_kernel_configuration(params, &grid_dim, &block_dim);

    unsigned int* h_B = (unsigned int*) malloc(SIGMA * sizeof(unsigned int));
    unsigned int* d_B;
    cudaMalloc((void**)&d_B, SIGMA * sizeof(unsigned int));
    unsigned int* h_pos = (unsigned int*) malloc(SIGMA * sizeof(unsigned int));
    unsigned int* d_pos;
    cudaMalloc((void**)&d_pos, SIGMA * sizeof(unsigned int));

    // Setup: malloc > timer start > memset/memcpy
    wrapper_setup(params, &d_text, &d_pattern, &d_match);

    /* Preprocessing */
    TicTocTimer preprocess_timer_start = tic();

    unsigned int occ[SIGMA] = {0};
    int start = 0, len = 0, i, j;
    for (i=0, j=0; i<params.pattern_size; i++) {
      if (occ[params.pattern[i]]) {
	while(params.pattern[j]!=params.pattern[i]) {
	  occ[params.pattern[j]]=0;
	  j++;
	} 
	occ[params.pattern[j]]=0;
	j++;
      }
      occ[params.pattern[i]]=1;
      if (len < i-j+1 ) {
	len = i-j+1;
	start = j;
      }
    }

    for (i=0; i<SIGMA; i++) h_pos[i]=-1;
    for (i=0; i<len; i++) h_pos[params.pattern[start+i]]=i;
    
    //printf("%d / %d\n",m,len);
    //for (i=start; i<start+len; i++) printf("%d ",x[i]);
    //if (start+len<m) printf("[%d] ",x[start+len]);
    //printf("\n\n");
    double preprocess_duration = toc(&preprocess_timer_start) * 1000;
    
    cudaMemcpy((d_text + params.text_size), params.pattern,
               params.pattern_size * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, SIGMA * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos, h_pos, SIGMA * sizeof(unsigned int), cudaMemcpyHostToDevice);

    // Kernel run
    gpuErrchk( cudaEventRecord(kernel_time.start) );
    backward_snr_dawg_matching<<<grid_dim, block_dim>>>(
						d_text, params.text_size, d_pattern, params.pattern_size,
						d_B, d_pos, len, start, params.stride_length, d_match);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaEventRecord(kernel_time.stop) );

    search_info timers = {0};
    timers.preprocess_duration = preprocess_duration;
    // Teardown: copy match back > timer stop > free
    wrapper_teardown(params, &timers, d_text, d_pattern, d_match);
    // Free memory
    cudaFree(d_B);
    cudaFree(d_pos);
    free(h_B);
    free(h_pos);
    return timers;
}
