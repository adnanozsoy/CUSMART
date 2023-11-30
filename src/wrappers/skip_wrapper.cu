#include "skip_wrapper.h"
#include "algos/skip.cuh"
#include "wrapper_helpers.h"
#include "util/cutil.cuh"
#include "util/tictoc.h"

#include <string.h>
#include <stdlib.h>


search_info skip_search_wrapper(search_parameters params){

    cuda_time kernel_time = {0}, total_time = {0};
    unsigned char *d_text, *d_pattern;
    int *d_match;
    unsigned int grid_dim, block_dim;

    setup_timers(&kernel_time, &total_time);
    get_kernel_configuration(params, &grid_dim, &block_dim);

    // Allocate memory for variables
    List *d_z;
    cudaMalloc((void**)&d_z, SIGMA * sizeof(List));
    List d_cells;
    cudaMalloc((void**)&d_cells, params.pattern_size * sizeof(struct _cell));
    List *h_z = (List*)calloc(SIGMA, sizeof(List));
    List h_cells = (List)calloc(params.pattern_size, sizeof(struct _cell));
	
    // Setup: malloc > timer start > memset/memcpy
    wrapper_setup(params, &d_text, &d_pattern, &d_match);

    // Preprocessing
    TicTocTimer preprocess_timer_start = tic();
    for (int i = 0; i < params.pattern_size; ++i) {
        List h_ptr = h_cells + i;
        h_ptr->element = i;
        h_ptr->next = h_z[params.pattern[i]];
        h_z[params.pattern[i]] = h_ptr;
    }
            
    // map pointers to device side
    for (int i = 0; i < params.pattern_size; ++i) {
        List h_ptr = h_cells + i;
        if(h_ptr->next)
            h_ptr->next = (h_ptr->next - h_cells) + d_cells;

    }
    for (int i = 0; i < SIGMA; ++i)
        if (h_z[i]) h_z[i] = (h_z[i] - h_cells) + d_cells;
    double preprocess_duration = toc(&preprocess_timer_start) * 1000;
    
    // move altered variables from host to device
    gpuErrchk( cudaMemcpy(d_z, h_z, SIGMA * sizeof(List), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_cells, h_cells, 
        params.pattern_size * sizeof(struct _cell), cudaMemcpyHostToDevice) );


    //Kernel run
    gpuErrchk( cudaEventRecord(kernel_time.start) );
    skip_search<<<grid_dim, block_dim>>>(
        d_text, params.text_size, d_pattern, params.pattern_size,
        d_z, params.stride_length, d_match);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaEventRecord(kernel_time.stop) );

	search_info timers = {0};
	timers.preprocess_duration = preprocess_duration;
    // Teardown: copy match back > timer stop > free
    wrapper_teardown(params, &timers, d_text, d_pattern, d_match);

	// Release memory
    gpuErrchk( cudaFree(d_cells) );
    free(h_cells);

    return timers;
}
