#ifndef CUTIL_CUH
#define CUTIL_CUH

#include <stdio.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
		fprintf(stderr,"[GPUassert] %s(%d): %s %s %d\n", 
			cudaGetErrorName (code), code, cudaGetErrorString(code), file, line);
		if (abort) exit(code);
   }
}

inline unsigned long divUp(unsigned long x, unsigned long y){ return (x + y-1) / y;}

#endif
