
#include "bom_wrapper.h"
#include "algos/bom.cuh"
#include "wrapper_helpers.h"
#include "util/cutil.cuh"
#include "util/tictoc.h"

#include <stdlib.h>
#include <stdio.h>


int get_index(List L[], List next_cell, int m){
	int index = 0;
	for(int i = 0; i <= m; i++){		
		List node = L[i];
		while(node!=NULL){			
			if(node == next_cell){
				//printf("adres %d was hitted with index %d \n", node, index);
				return index;
			}
			else
				node = node->next;
			
			index++;		
		}
		
		if(L[i]==NULL)	
			index++;
	}
	
	//printf("next address was not found %d \n", next_cell);
	return -1;
}

struct element_index * list2arr(List L[], int m, int *list2array_length){
	int len = 0;
	for(int i = 0; i <= m; i++){		
		List j = L[i];
		while(j!=NULL){	
			j = j->next;		
			len++;			
		}
		
		if(L[i]==NULL)	
			len++;
	}
	*list2array_length = len;
	
	//printf("list2array size %d, %d \n", len, *list2array_length);
	
	element_index *L2 = (element_index *)malloc((*list2array_length) * sizeof(element_index));
	int index = 0;
	for(int i = 0; i <= m; i++){
		List j = L[i];
		while(j!=NULL){
			L2[index].element = j->element;
			L2[index].next_index = get_index(L, j->next, m);
			j = j->next;			
			//printf("%d %d %d %d\n\n", i, index, L2[index].element, L2[index].next_index);
			//printf("i= %d, index=%d, element= %d, next index=%d adres is %d \n", i, index, L2[index].element, L2[index].next_index, j->next);
			index++;
		}
		
		if(L[i]==NULL)	{	
			//printf("blank cell %d index is %d\n", j, i);		
			L2[index].element = -1;	
			L2[index].next_index = -1;
			index++;
		}
	}
	return L2;
}

search_info backward_oracle_matching_wrapper(search_parameters params){

	cuda_time kernel_time = {0}, total_time = {0};
	unsigned char *d_text, *d_pattern;
	int *d_match;
	unsigned int grid_dim, block_dim;
	
	setup_timers(&kernel_time, &total_time);
	get_kernel_configuration(params, &grid_dim, &block_dim);

	/* Preprocessing */
	TicTocTimer preprocess_timer_start = tic();
	char T[XSIZE + 1];
	List L[XSIZE + 1];  
	memset(L, 0, (params.pattern_size + 1)*sizeof(List));
	memset(T, FALSE, (params.pattern_size + 1)*sizeof(char));
	oracle(params.pattern, params.pattern_size, T, L);
	
	int *list2array_length = (int *)malloc(sizeof(int));
	element_index *L2 = list2arr(L, params.pattern_size, list2array_length);
	
	char *d_T;
	element_index *d_L;
	double preprocess_duration = toc(&preprocess_timer_start) * 1000;
	
	gpuErrchk( cudaMalloc(&d_T, (params.pattern_size + 1)*sizeof(char)) );
	gpuErrchk( cudaMalloc(&d_L, (*list2array_length) * sizeof(struct element_index)) );
	
	
	// Setup: malloc > timer > memset/memcpy
	wrapper_setup(params, &d_text, &d_pattern, &d_match);

	gpuErrchk( cudaMemcpy(d_T, T, (params.pattern_size + 1)*sizeof(char), cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(d_L, L2, (*list2array_length) * sizeof(struct element_index), cudaMemcpyHostToDevice) );
		
	// Kernel run
	gpuErrchk( cudaEventRecord(kernel_time.start) );
	backward_oracle_matching<<<grid_dim, block_dim>>>(d_text, params.text_size, d_pattern, params.pattern_size, d_L, d_T, params.stride_length, d_match);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaEventRecord(kernel_time.stop) );

	search_info timers = {0};
	timers.preprocess_duration = preprocess_duration;
	// Teardown: copy match back > timer stop > free
	wrapper_teardown(params, &timers, d_text, d_pattern, d_match);
	// Release memory
	gpuErrchk( cudaFree(d_T) );
	gpuErrchk( cudaFree(d_L) );
	free(list2array_length);
	for (int i=0; i<=params.pattern_size; i++) free(L[i]);

	return timers;
}
