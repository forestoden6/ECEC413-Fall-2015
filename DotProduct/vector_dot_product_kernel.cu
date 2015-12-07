#ifndef _VECTOR_DOT_PRODUCT_KERNEL_H_
#define _VECTOR_DOT_PRODUCT_KERNEL_H_

#define BLOCK_SIZE 1024
#define GRID_SIZE 1024

/* Edit this function to complete the functionality of dot product on the GPU using atomics. 
	You may add other kernel functions as you deem necessary. 
 */

__device__ void lock(int *mutex);
__device__ void unlock(int *mutex); 
 
__global__ void vector_dot_product_kernel(int num_elements, float* a, float* b, float* result, int *mutex)
{

	__shared__ float runningSums[BLOCK_SIZE];
	
	int tx = threadIdx.x;
	int threadID = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	float local_thread_sum = 0.0f;
	unsigned int i = threadID;

	/* Allows few threadblocks to span the entire vector, decreasing overhead */
	while( i < num_elements ) {

		/* take the dot product and stride across vector */
		local_thread_sum += a[i] * b[i];
		i += stride;
	}

	/* Use shared memory to do reduction in each TB */
	runningSums[threadIdx.x] = local_thread_sum;
	__syncthreads();
	
	/* First half of the threadblock will take the if
	 * then first quarter, etc. Eventually one thread will 
	 * add the last two partial sums together */
	for(int stride = blockDim.x/2; stride > 0; stride /= 2 )
	{
		if(tx < stride)
			runningSums[tx] += runningSums[tx+stride];
		__syncthreads();
	}
	
	/* One thread in each TB will acquire the lock
	 * and add its block's partial sum to the global 
	 * dot product result */
	if (threadIdx.x == 0) {
		lock(mutex);
		result[0] += runningSums[0] ;
		unlock(mutex);
	}

}

/* Using CAS to acquire mutex. */
__device__ void lock(int *mutex)
{
  while(atomicCAS(mutex, 0, 1) != 0);
}

/* Using exchange to release mutex. */
__device__ void unlock(int *mutex)
{
  atomicExch(mutex, 0);
}



#endif // #ifndef _VECTOR_DOT_PRODUCT_KERNEL_H
