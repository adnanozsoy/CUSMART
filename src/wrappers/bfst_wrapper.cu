
#include "bfst_wrapper.h"
#include "algos/bf.cuh"
#include "wrapper_helpers.h"
#include "util/cutil.cuh"
#include "util/reduction.cuh"

#include <stdio.h>

search_info brute_force_stream_wrapper(search_parameters params){

    cuda_time total_time = {0};
    unsigned char *d_text, *d_pattern;
    int *d_match;
    params.stride_length = 1;

    const int num_streams = 64;
    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; i++)
        cudaStreamCreate(&streams[i]);

    int stream_blocksize = divUp(params.text_size, num_streams);

    gpuErrchk( cudaEventCreate(&total_time.start) );
    gpuErrchk( cudaEventCreate(&total_time.stop) );

    // Setup: malloc > timer start > memset/memcpy
    // wrapper_setup(params, &d_text, &d_pattern, &d_match);

    // wrapper setup start
    size_t text_alloc_size = (params.text_size+params.pattern_size+1) * sizeof(unsigned char);

    gpuErrchk( cudaMalloc((void**)&d_text,   text_alloc_size));
    gpuErrchk( cudaMalloc((void**)&d_match,   params.text_size * sizeof(int)) );
    gpuErrchk( cudaMemset(d_match, 0, params.text_size * sizeof(int)) );

    if (!params.constant_memory)
        gpuErrchk( cudaMalloc((void**)&d_pattern, params.pattern_size * sizeof(unsigned char)) );

    unsigned char* h_text = params.text;
    unsigned char *pinned_text;
    if (params.pinned_memory)
    {
        gpuErrchk( cudaMallocHost((void**)&pinned_text, text_alloc_size) );
        memcpy(pinned_text, params.text, params.text_size * sizeof(unsigned char));
        h_text = pinned_text;
    }

    gpuErrchk( cudaEventRecord(total_time.start) );

    gpuErrchk( cudaMemcpy(d_pattern, params.pattern, 
        params.pattern_size * sizeof(unsigned char), cudaMemcpyHostToDevice) );

    // wrapper setup end

    for (int i = 0; i < num_streams; i++){

        int stream_currblocksize;
        if ((i+1)*stream_blocksize + params.pattern_size <= params.text_size)
            stream_currblocksize = stream_blocksize + params.pattern_size;
        else
            stream_currblocksize = params.text_size - i*stream_blocksize;

        // printf("I:%2d - NumStreams: %d blocksize: %d curblocksize: %d start %d total %d\n",
        //     i, num_streams, stream_blocksize, stream_currblocksize, (i*stream_blocksize), params.text_size);

        unsigned int block_count = divUp(stream_currblocksize, params.stride_length);
        unsigned int block_dim = block_count > params.block_dim ? params.block_dim : divUp(block_count, 32) * 32;
        unsigned int grid_dim = divUp(stream_currblocksize, block_dim * params.stride_length);

        // transfer text to device partially
        gpuErrchk( cudaMemcpyAsync(d_text+(i*stream_blocksize), h_text+(i*stream_blocksize), 
            stream_currblocksize * sizeof(unsigned char), cudaMemcpyHostToDevice, streams[i]) );
        // kernel run
        brute_force<<<grid_dim, block_dim, 0, streams[i]>>>(
            d_text+(i*stream_blocksize), stream_currblocksize, 
            d_pattern, params.pattern_size, 
            d_match+(i*stream_blocksize));
        // gpuErrchk( cudaPeekAtLastError() );        
    }
    search_info timers = {0};
    // Teardown: copy match back > timer stop > free

    // Wrapper Teardown start

    cuda_time reduction_time = {0};
    int* d_match_count;
    if (params.gpu_reduction){
        gpuErrchk( cudaEventCreate(&reduction_time.start) );
        gpuErrchk( cudaMalloc((void**)&d_match_count, sizeof(int)) );
        device_reduce_block_atomic(d_match, d_match_count, params.text_size);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaMemcpy(params.match, d_match_count, sizeof(int), cudaMemcpyDeviceToHost) );
    }
    else {
        gpuErrchk( cudaMemcpy(params.match, d_match, 
            params.text_size * sizeof(int), cudaMemcpyDeviceToHost) );
    }
    gpuErrchk( cudaEventRecord(total_time.stop) );
    gpuErrchk( cudaEventSynchronize(total_time.stop) );
    gpuErrchk( cudaEventElapsedTime(&timers.total_duration, total_time.start, total_time.stop) );
    gpuErrchk( cudaEventDestroy(total_time.start) );
    gpuErrchk( cudaEventDestroy(total_time.stop) );

    gpuErrchk( cudaFree(d_text) );
    gpuErrchk( cudaFree(d_match) );
    if (!params.constant_memory) gpuErrchk( cudaFree(d_pattern) );
    if (params.gpu_reduction)    gpuErrchk( cudaFree(d_match_count) );
    if (params.pinned_memory)    gpuErrchk( cudaFreeHost(pinned_text) );

    // Wrapper Teardown end
    // wrapper_teardown(params, &timers, d_text, d_pattern, d_match);
    for (int i = 0; i < num_streams; i++)
        cudaStreamDestroy(streams[i]);
    return timers;
}
