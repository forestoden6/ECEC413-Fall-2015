#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>

#define BLOCK_SIZE 16

__device__ void lock(int *mutex);
__device__ void unlock(int *mutex); 

texture<float>inputTex1D;
texture<float>outputTex1D;
texture<float>diffTex1D;

__global__ void 
solver_kernel_naive(float* input, float* output, int N, float* globalDiff){
	
	unsigned int tx = threadIdx.x;
	unsigned int x = blockIdx.x * blockDim.x + tx;
	
	unsigned int ty = threadIdx.y;
	unsigned int y = blockIdx.y * blockDim.y + ty;
		
	if(x > 0 && y > 0 && x < (N-1) && y < (N-1)) {
		output[x*N + y] = 0.20f * (input[x*N + y] + input[(x-1)*N +y] + input[(x+1)*N +y] +\
			input[x*N + (y-1)] + input[x*N + (y+1)]);
	}
	else
		output[x*N+y] = input[x*N+y];
			
	globalDiff[x*N+y] = fabsf(output[x*N + y] - input[x*N + y]); 
}

__global__ void 
solver_kernel_optimized(float* input, float* output, int N, float* globalDiff){
	
	unsigned int tx = threadIdx.x;
	unsigned int x = blockIdx.x * blockDim.x + tx;
	
	unsigned int ty = threadIdx.y;
	unsigned int y = blockIdx.y * blockDim.y + ty;
		
	if(x > 0 && y > 0 && x < (N-1) && y < (N-1)) {
		output[x*N + y] = 0.20f * (tex1Dfetch(inputTex1D,x*N + y) + tex1Dfetch(inputTex1D,(x-1)*N +y) + tex1Dfetch(inputTex1D,(x+1)*N +y) +\
			tex1Dfetch(inputTex1D,x*N + (y-1)) + tex1Dfetch(inputTex1D,x*N + (y+1)));
	}
	else
		output[x*N+y] = tex1Dfetch(inputTex1D,x*N+y);
			
	globalDiff[x*N+y] = fabsf(tex1Dfetch(outputTex1D,x*N + y) - tex1Dfetch(inputTex1D,x*N + y)); 
}


#endif /* _MATRIXMUL_KERNEL_H_ */
