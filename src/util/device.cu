
#include "device.h"
#include "cuda_runtime.h"


void device_reset()
{
	cudaDeviceReset();
}