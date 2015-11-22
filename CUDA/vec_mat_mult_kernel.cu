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
	
	//Block Index
	int blockX = blockIdx.x;
	
	//Absolute Position
	int row = blockDim.x * blockX + threadX;
		
	double Y_temp = 0;
	float A_element = 0;
	float X_element = 0;
	for(int i = 0; i < MATRIX_SIZE; i++){
		A_element = Ad[MATRIX_SIZE * row + i]; //Get all the values in the row of A
		X_element = Xd[i]; //Get all values in X
		Y_temp += A_element * X_element;
	}
	
	Yd[row] = (float)Y_temp;
}


__global__ void vec_mat_kernel_optimized(float *Ad, float *Xd, float *Yd)
{
	__shared__ float Asub[TILE_SIZE][TILE_SIZE];
	__shared__ float Xsub[TILE_SIZE];
	
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	
	int row = blockDim.y * blockIdx.y + ty;
	int col = blockDim.x * blockIdx.x + tx;
	
	int k = 0;
	int temp;
	double Ysub = 0.0f;
	
	while(k < MATRIX_SIZE){
		if(k + ty < MATRIX_SIZE && col < MATRIX_SIZE)
			Asub[ty][tx] = Ad[row*MATRIX_SIZE + k + tx];
		
		if(ty == 0)
			Xsub[tx] = Xd[(k+ty)+col];
		
		__syncthreads();
		
		if(ty == 0)
			for(temp = 0; temp < TILE_SIZE; temp++)
				Ysub += Asub[tx][temp] * Xsub[temp];
			
		__syncthreads();
		
		k += TILE_SIZE;
	}
	if(ty == 0)
		Yd[row + col] = (float)Ysub;
}



#endif // #ifndef _MATRIXMUL_KERNEL_H_
