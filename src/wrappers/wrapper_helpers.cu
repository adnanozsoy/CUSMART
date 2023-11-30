
#include "wrapper_helpers.h"
#include "util/cutil.cuh"
#include "util/reduction.cuh"
#include "cuda_runtime.h"

static cuda_time *_total;
static cuda_time *_kernel;
static cuda_time *_reduce;
static cuda_time *_setupcopy;
static unsigned char *pinned_text;

__device__ int warmup_memory = 0;
__global__ void warmup_kernel(){ warmup_memory ^= 1; }

__constant__ unsigned char cpattern[32768];

void wrapper_setup(
	search_parameters p, unsigned char **d_text, unsigned char **d_pattern, int **d_match)
{
	wrapper_setup_malloc(p, d_text, d_pattern, d_match);
    // Warmup kernel to filter out startup overhead
    warmup_kernel<<<1, 1>>>();

	gpuErrchk( cudaEventRecord(_total->start) );
	gpuErrchk( cudaEventRecord(_setupcopy->start) );
	wrapper_setup_memcpy(p, d_text, d_pattern, d_match);
	gpuErrchk( cudaEventRecord(_setupcopy->stop) );
}

void wrapper_setup_malloc(
	search_parameters p, unsigned char **d_text, unsigned char **d_pattern, int **d_match)
{
	size_t text_alloc_size = (p.text_size+p.pattern_size+1) * sizeof(unsigned char);

	gpuErrchk( cudaMalloc((void**)d_text, 	 text_alloc_size));
	gpuErrchk( cudaMalloc((void**)d_match,   p.text_size * sizeof(int)) );
	if (!p.constant_memory)
		gpuErrchk( cudaMalloc((void**)d_pattern, p.pattern_size * sizeof(unsigned char)) );

	if (p.pinned_memory) {
		gpuErrchk( cudaMallocHost((void**)&pinned_text, text_alloc_size) );
		memcpy(pinned_text, p.text, p.text_size * sizeof(unsigned char));
	}
}

void wrapper_setup_memcpy(
	search_parameters p, unsigned char **d_text, unsigned char **d_pattern, int **d_match)
{
	unsigned char* text_ptr;
	if (p.pinned_memory)
		text_ptr = pinned_text;
	else
		text_ptr = p.text;

	gpuErrchk( cudaMemcpy(*d_text, text_ptr, 
		p.text_size * sizeof(unsigned char), cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemset(*d_match, 0, p.text_size * sizeof(int)) );
	if (p.constant_memory) {
		gpuErrchk( cudaMemcpyToSymbol(cpattern, p.pattern, p.pattern_size * sizeof(unsigned char), 0, cudaMemcpyHostToDevice) );
		gpuErrchk( cudaGetSymbolAddress((void**)d_pattern, cpattern) );
	} else {
		gpuErrchk( cudaMemcpy(*d_pattern, p.pattern, 
		p.pattern_size * sizeof(unsigned char), cudaMemcpyHostToDevice) );
	}	
}

void wrapper_teardown(
	search_parameters p, search_info *timers, 
	unsigned char *d_text, unsigned char *d_pattern, int *d_match)
{
	int* d_match_count;
	if (p.gpu_reduction){
		gpuErrchk( cudaMalloc((void**)&d_match_count, sizeof(int)) );
		gpuErrchk( cudaEventRecord(_reduce->start) );
		device_reduce_block_atomic(d_match, d_match_count, p.text_size);
		gpuErrchk( cudaPeekAtLastError() );
		gpuErrchk( cudaMemcpy(p.match, d_match_count, sizeof(int), cudaMemcpyDeviceToHost) );
		gpuErrchk( cudaEventRecord(_reduce->stop) );
	}
	else {
		gpuErrchk( cudaMemcpy(p.match, d_match, 
			p.text_size * sizeof(int), cudaMemcpyDeviceToHost) );
	}

	gpuErrchk( cudaEventRecord(_total->stop) );

	gpuErrchk( cudaFree(d_text) );
	gpuErrchk( cudaFree(d_match) );
	if (!p.constant_memory)
		gpuErrchk( cudaFree(d_pattern) );
	if (p.gpu_reduction) {
		gpuErrchk( cudaFree(d_match_count) );
	}
	if (p.pinned_memory) {
		gpuErrchk( cudaFreeHost(pinned_text) );
	}

	gpuErrchk( cudaEventSynchronize(_total->stop) );
	cudaError_t code;
	code = cudaEventElapsedTime(
		&(timers->kernel_duration), _kernel->start, _kernel->stop);
	if (code == cudaErrorInvalidResourceHandle) timers->kernel_duration = 0;
	code = cudaEventElapsedTime(
		&(timers->total_duration), _total->start, _total->stop);
	if (code == cudaErrorInvalidResourceHandle) timers->total_duration = 0;
	code = cudaEventElapsedTime(
		&(timers->setupcopy_duration), _setupcopy->start, _setupcopy->stop);
	if (code == cudaErrorInvalidResourceHandle) timers->setupcopy_duration = 0;
	if (p.gpu_reduction){
		gpuErrchk( cudaEventElapsedTime(
			&(timers->reduce_duration), _reduce->start, _reduce->stop) );
	}

	gpuErrchk( cudaEventDestroy(_total->start) );
	gpuErrchk( cudaEventDestroy(_total->stop) );
	gpuErrchk( cudaEventDestroy(_kernel->start) );
	gpuErrchk( cudaEventDestroy(_kernel->stop) );
	gpuErrchk( cudaEventDestroy(_reduce->start) );
	gpuErrchk( cudaEventDestroy(_reduce->stop) );
	gpuErrchk( cudaEventDestroy(_setupcopy->start) );
	gpuErrchk( cudaEventDestroy(_setupcopy->stop) );

	free(_reduce);
	free(_setupcopy);
}

void get_kernel_configuration(
	search_parameters p, unsigned int *grid_dim, unsigned int *block_dim)
{
	get_kernel_configuration_shared(p, 0, grid_dim, block_dim);
}

void get_kernel_configuration_shared(
	search_parameters p, int shared_size,
	unsigned int *grid_dim, unsigned int *block_dim)
{
	int block_count = divUp(p.text_size, p.stride_length);

	if (shared_size != 0){
		int block_dim_max = 
			((shared_size - p.pattern_size) / p.stride_length / 32) * 32;
		if (block_dim_max > block_count)
			*block_dim = min( (int)divUp(block_count, 32) * 32, p.block_dim );
		else
			*block_dim = min( block_dim_max, p.block_dim );
	}
	else
		*block_dim = block_count > p.block_dim ?
		 p.block_dim : divUp(block_count, 32) * 32;


	*grid_dim = divUp(p.text_size, (*block_dim) * p.stride_length);
}

void setup_timers(cuda_time *kernel, cuda_time *total)
{
	_total = total;
	_kernel = kernel;
	_reduce = (struct cuda_time*)malloc(sizeof(struct cuda_time));
	_setupcopy = (struct cuda_time*)malloc(sizeof(struct cuda_time));
	gpuErrchk( cudaEventCreate(&(_total->start)) );
	gpuErrchk( cudaEventCreate(&(_total->stop)) );
	gpuErrchk( cudaEventCreate(&(_kernel->start)) );
	gpuErrchk( cudaEventCreate(&(_kernel->stop)) );
	gpuErrchk( cudaEventCreate(&(_reduce->start)) );
	gpuErrchk( cudaEventCreate(&(_reduce->stop)) );
	gpuErrchk( cudaEventCreate(&(_setupcopy->start)) );
	gpuErrchk( cudaEventCreate(&(_setupcopy->stop)) );
}
