#include "reduction.cuh"
#include "util/cutil.cuh"

__inline__ __device__
int warpReduceSum(int val) {
  for (int offset = warpSize/2; offset > 0; offset /= 2)
    val += __shfl_down(val,offset);
  return val;
}

__inline__ __device__
int blockReduceSum(int val) {
  static __shared__ int shared[32];
  int lane=threadIdx.x%warpSize;
  int wid=threadIdx.x/warpSize;
  val=warpReduceSum(val);

  //write reduced value to shared memory
  if(lane==0) shared[wid]=val;
  __syncthreads();

  //ensure we only grab a value from shared memory if that warp existed
  val = (threadIdx.x<blockDim.x/warpSize) ? shared[lane] : int(0);
  if(wid==0) val=warpReduceSum(val);

  return val;
}

__global__ void device_reduce_block_atomic_kernel(int *in, int* out, int N) {
  int sum=int(0);
  for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<N;i+=blockDim.x*gridDim.x) {
    sum+=in[i];
  }
  sum=blockReduceSum(sum);
  if(threadIdx.x==0)
    atomicAdd(out,sum);
}

void device_reduce_block_atomic(int *in, int* out, int N) {
  int threads=256;
  int blocks=min((N+threads-1)/threads,2048);

  cudaMemsetAsync(out,0,sizeof(int));
  device_reduce_block_atomic_kernel<<<blocks,threads>>>(in,out,N);
  gpuErrchk( cudaPeekAtLastError() );
}