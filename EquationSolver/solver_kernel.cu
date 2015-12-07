#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>

#define BLOCK_SIZE 16

texture<float, 2>inputTex2D;

__global__ void 
solver_kernel_naive(float* input, float* output, int N, float* globalDiff){

	//__shared__ float runningSums[BLOCK_SIZE*BLOCK_SIZE];
	
	unsigned int tx = threadIdx.x;
	unsigned int x = blockIdx.x * blockDim.x + tx;
	
	unsigned int ty = threadIdx.y;
	unsigned int y = blockIdx.y * blockDim.y + ty;
		
	if(x > 0 && y > 0 && x < (N-1) && y < (N-1))
		output[x*N + y] = 0.20f * (input[x*N + y] + input[(x-1)*N +y] + input[(x+1)*N +y] +\
			input[x*N + (y-1)] + input[x*N + (y+1)]);
			
	
	globalDiff[x*N+y] = fabsf(output[x*N + y] - input[x*N + y]); 
	
	__syncthreads();
	
	for(int stride = (blockDim.x*blockDim.y)/2; stride > 0; stride /= 2)
	{
		if(x*N+y < stride)
			globalDiff[x*N+y] += globalDiff[x*N+y+stride];
		__syncthreads();
	}
}

__global__ void 
solver_kernel_optimized(float* input, float* output, int N, float* globalDiff){
	
	unsigned int tx = threadIdx.x;
	unsigned int x = blockIdx.x * blockDim.x + tx;
	
	unsigned int ty = threadIdx.y;
	unsigned int y = blockIdx.y * blockDim.y + ty;
		
	if(x > 0 && y > 0 && x < (N-1) && y < (N-1))
		output[x*N + y] = 0.20f * (input[x*N + y] + input[(x-1)*N +y] + input[(x+1)*N +y] +\
			input[x*N + (y-1)] + input[x*N + (y+1)]);
			
	
	globalDiff[x*N+y] = fabsf(output[x*N + y] - input[x*N + y]); 
	
	__syncthreads();
	
	for(int stride = (blockDim.x*blockDim.y)/2; stride > 0; stride /= 2)
	{
		if(x*N+y < stride)
			globalDiff[x*N+y] += globalDiff[x*N+y+stride];
		__syncthreads();
	}
}

#endif /* _MATRIXMUL_KERNEL_H_ */