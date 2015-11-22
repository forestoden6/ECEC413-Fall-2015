
/* Matrix multiplication: C = A * B.
 * Device code.

	Author: Naga Kandasamy
	Date modified: 11/05/2015
 */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>
#include "matrixmul.h"

__global__ void 
MatrixMulKernel(Matrix M, Matrix N, Matrix P)
{
    __shared__ float Msub[TILE_SIZE][TILE_SIZE];
    __shared__ float Nsub[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x; // Obtain the x-index within the thread block
    int ty = threadIdx.y; // Obtain the y-index within the thread block
    int row = (blockDim.y * blockIdx.y + ty); // Perform the thread to data ID mapping
    int col = blockDim.x * blockIdx.x + tx;
    int k = 0;
    int temp;
    double Psub = 0.0f;
  
    while(k < M.width){
        // Check M edge condtions for this tile
        if(k + tx < M.width && row < M.height)
            Msub[ty][tx] = M.elements[row*M.width + k + tx];
        else
            Msub[ty][tx] = 0.0f; // Pad out the shared memory area 

    
        // Check N edge conditions for this tile
        if(k + threadIdx.y < N.height && col < N.width)
            Nsub[ty][tx] = N.elements[(k+ty)*N.width + col];
        else
            Nsub[ty][tx] = 0.0f; // Pad out the shared memory area

        __syncthreads(); // Barrier sync for threads to wait while shared memory is populated by the thread block

    
        // Multiply the row and column entries corresponding to the tile just loaded 
        for(temp = 0; temp < TILE_SIZE; temp++)
            Psub += Msub[ty][temp] * Nsub[temp][tx];

        __syncthreads();
    
        k += TILE_SIZE;
  }

    // Output edge condition check
    if(col < P.width && row < P.height)
        P.elements[row*P.width + col] = (float)Psub;

    return;

}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
