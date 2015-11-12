/* Vector-Matrix multiplication: Y = A * X.
 * Device code.
 */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include "vec_mat_mult.h"

__global__ void vec_mat_kernel_naive(float *Ad, float *Xd, float *Yd)
{
	//Thread Index
	int threadX = threadIdx.x;
	//int threadY = threadIdx.y;
	
	//Block Index
	int blockX = blockIdx.x;
	//int blockY = blockIdx.y;
	
	//Find absolute position
	int col = blockDim.x * blockX + threadX;
	//int row = blockDim.y * blockY + threadY;
	
	double Y_temp = 0;
	for(int i = 0; i < MATRIX_SIZE; i++){
		double A_element = Ad[MATRIX_SIZE * col + i]; //Get all the values in the row of A
		double X_element = Xd[i]; //Get all values in X
		Y_temp += A_element * X_element;
	}
	
	Yd[col] = (float)Y_temp;
}


__global__ void vec_mat_kernel_optimized(float *Ad, float *Xd, float *Yd)
{
	__shared__ float A_tile[TILE_SIZE][TILE_SIZE];
	__shared__ float X_tile[TILE_SIZE];
	
	//Thread Index
	int threadX = threadIdx.x;
	int threadY = threadIdx.y;
	
	//Block Index
	int blockX = blockIdx.x;
	int blockY = blockIdx.y;
	
	//Find absolute position
	int col = blockDim.x * blockX + threadX;
	int row = blockDim.y * blockY + threadY;
	
	int i = 0;
	double Y_temp = 0;
	
	while(i < MATRIX_SIZE){
		if(i + threadX < MATRIX_SIZE && row < MATRIX_SIZE)
			A_tile[threadY][threadX] = Ad[row * MATRIX_SIZE + i + threadX];
		else
			A_tile[threadY][threadX] = 0.0f;
			
		if(threadX < 1 && col < MATRIX_SIZE)
			X_tile[threadY] = Xd[(i+threadY)*MATRIX_SIZE + col];
		/*else
			X_tile[threadY][threadX] = 0.0f;*/
			
		__syncthreads();
		
		if(threadX == 0)
			for(int temp = 0; temp < TILE_SIZE; temp++)
				Y_temp += A_tile[threadY][temp] * X_tile[temp];
			
		__syncthreads();
		
		i += TILE_SIZE;
	}

	if(col < 1 && row < MATRIX_SIZE)
		Yd[row] = (float)Y_temp;
}



#endif // #ifndef _MATRIXMUL_KERNEL_H_
