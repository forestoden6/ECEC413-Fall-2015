 /* Device code. */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include "gauss_eliminate.h"

__global__ void gauss_eliminate_kernel(const float* A, float *U, int current_row, int k, int offset)
{
	int tx = threadIdx.x;

	int row = current_row * MATRIX_SIZE;
	
	if(tx > k)
	{
		//Division Step
		U[tx + row] = (float) A[tx + row] / (float) A[row + k];
	}
	
	__syncthreads();
	
	if(tx == k)
		U[tx + row] = 1;
		
	
	if(tx < k)
	{
		U[tx + row] = 0;
	}
	else
	{ 
		//Elimination Step
		for(int i = (k+1); i < MATRIX_SIZE; i++)
		{
			U[tx + row +(i* MATRIX_SIZE)] = U[tx + row + (i* MATRIX_SIZE)] - (U[k + row +(i* MATRIX_SIZE)] * U[(k * MATRIX_SIZE) + tx]);
		}
	}
	
}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
