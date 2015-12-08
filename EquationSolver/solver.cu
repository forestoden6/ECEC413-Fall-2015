/* 
Code for the equation solver. 
Author: Forest Oden 
Date: 12/7/2015 
*/

#include <stdio.h>
#include <string.h>
#include <malloc.h>
#include <sys/time.h>
#include <stdlib.h>
#include <math.h>
#include "grid.h" // This file defines the grid data structure

// includes, kernels
#include "solver_kernel.cu"

void check_for_error(char *);
extern "C" void compute_gold(GRID_STRUCT *);


/* This function prints the grid on the screen */
void 
display_grid(GRID_STRUCT *my_grid)
{
	for(int i = 0; i < my_grid->dimension; i++)
		for(int j = 0; j < my_grid->dimension; j++)
			printf("%f \t", my_grid->element[i * my_grid->dimension + j]);
   		
		printf("\n");
}


/* This function prints out statistics for the converged values, including min, max, and average. */
void 
print_statistics(GRID_STRUCT *my_grid)
{
		// Print statistics for the CPU grid
		float min = INFINITY;
		float max = 0.0;
		double sum = 0.0; 
		for(int i = 0; i < my_grid->dimension; i++){
			for(int j = 0; j < my_grid->dimension; j++){
				sum += my_grid->element[i * my_grid->dimension + j]; // Compute the sum
				if(my_grid->element[i * my_grid->dimension + j] > max) max = my_grid->element[i * my_grid->dimension + j]; // Determine max
				if(my_grid->element[i * my_grid->dimension + j] < min) min = my_grid->element[i * my_grid->dimension + j]; // Determine min
				 
			}
		}

	printf("AVG: %f \n", sum/(float)my_grid->num_elements);
	printf("MIN: %f \n", min);
	printf("MAX: %f \n", max);

	printf("\n");
}


/* Calculate the differences between grid elements for the various implementations. */
void compute_grid_differences(GRID_STRUCT *grid_1, GRID_STRUCT *grid_2)
{
    float diff;
    int dimension = grid_1->dimension;
    int num_elements = dimension*dimension;

    diff = 0.0;
    for(int i = 0; i < grid_1->dimension; i++){
        for(int j = 0; j < grid_1->dimension; j++){
            diff += fabsf(grid_1->element[i * dimension + j] - grid_2->element[i * dimension + j]);
        }
    }
    printf("Average difference in grid elements for Gauss Seidel and Jacobi methods = %f. \n", \
            diff/num_elements);

}



/* This function creates a grid of random floating point values bounded by UPPER_BOUND_ON_GRID_VALUE */
void 
create_grids(GRID_STRUCT *grid_for_cpu, GRID_STRUCT *grid_for_gpu)
{
	printf("Creating a grid of dimension %d x %d. \n", grid_for_cpu->dimension, grid_for_cpu->dimension);
	grid_for_cpu->element = (float *)malloc(sizeof(float) * grid_for_cpu->num_elements);
	grid_for_gpu->element = (float *)malloc(sizeof(float) * grid_for_gpu->num_elements);


	srand((unsigned)time(NULL)); // Seed the the random number generator 
	
	float val;
	for(int i = 0; i < grid_for_cpu->dimension; i++)
		for(int j = 0; j < grid_for_cpu->dimension; j++){
			val =  ((float)rand()/(float)RAND_MAX) * UPPER_BOUND_ON_GRID_VALUE; // Obtain a random value
			grid_for_cpu->element[i * grid_for_cpu->dimension + j] = val; 	
			grid_for_gpu->element[i * grid_for_gpu->dimension + j] = val; 				
		}
}


/* Edit this function skeleton to solve the equation on the device. Store the results back in the my_grid->element data structure for comparison with the CPU result. */
void 
compute_on_device(GRID_STRUCT *my_grid)
{

	int done = 0;
	int num_iter = 0;
	float diff = 0.0f;
	struct timeval start, stop;	
	float time = 0.0f;
	cudaError_t cudaStatus;
	
	float* grid2_h = (float *)malloc(sizeof(float) * my_grid->num_elements);
	float* grid2_d = NULL;
	cudaMalloc((void**) &grid2_d, sizeof(float) * my_grid->num_elements);
	cudaMemset(grid2_d, 0.0f, my_grid->num_elements);

	float* grid1_h = (float *)malloc(sizeof(float) * my_grid->num_elements);
	float* grid1_d = NULL;
	cudaMalloc((void**) &grid1_d, sizeof(float) * my_grid->num_elements);
	cudaMemcpy(grid1_d, my_grid->element, sizeof(float) * my_grid->num_elements, cudaMemcpyHostToDevice);
	
	float* diff_h = (float *)malloc(sizeof(float) * my_grid->num_elements);
	float* diff_d = NULL;
	cudaMalloc((void**) &diff_d, sizeof(float) * my_grid->num_elements);
	cudaMemset(diff_d, 0.0f, my_grid->num_elements);
	
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(GRID_DIMENSION/BLOCK_SIZE,GRID_DIMENSION/BLOCK_SIZE);
	/*
	/////////////////////////////////////////////////////////////////////
	//Using Global Memory
	/////////////////////////////////////////////////////////////////////

	printf("Solving using GPU Global Memory. \n");
	
	
	while(!done)
	{
		gettimeofday(&start, NULL);
		solver_kernel_naive<<< dimGrid, dimBlock >>>(grid1_d,grid2_d, GRID_DIMENSION, diff_d);
		cudaThreadSynchronize();

		gettimeofday(&stop, NULL);
		
		time += (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000);

		float* temp = grid1_d;
		grid1_d = grid2_d;
		grid2_d = temp;
		cudaMemcpy(diff_h, diff_d, sizeof(float)*my_grid->num_elements, cudaMemcpyDeviceToHost);
		gettimeofday(&start, NULL);
		for(int i = 0; i < my_grid->num_elements; i++)
			diff += diff_h[i];
		if(diff/(my_grid->num_elements) < (float) TOLERANCE)
			done = 1;
		gettimeofday(&stop, NULL);
		time += (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000);

		//printf("Difference: %f, Iteration: %d\n", diff/(my_grid->num_elements), num_iter);

		diff=0;
		num_iter++;
	}
	
	printf("Execution time = %fs. \n", time);
	
	check_for_error("KERNEL FAILURE");

	cudaStatus = cudaMemcpy(my_grid->element, grid1_d, sizeof(float) * my_grid->num_elements, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		const char *str = (char*) malloc(1024); // To store error string
		str = cudaGetErrorString(cudaStatus);
		fprintf(stderr, "CUDA Error!:: %s\n", str);
		getchar();
	}
	*/
	/////////////////////////////////////////////////////////////////////
	//Texture Memory
	/////////////////////////////////////////////////////////////////////
	
	cudaMemcpy(grid1_d, my_grid->element, sizeof(float) * my_grid->num_elements, cudaMemcpyHostToDevice);
	cudaMemset(grid2_d, 0.0f, my_grid->num_elements);
	cudaMemset(diff_d, 0.0f, my_grid->num_elements);
	memset(diff_h, 0.0f, my_grid->num_elements);
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
	cudaBindTexture2D(NULL, inputTex1D, grid1_d, desc, my_grid->dimension, my_grid->dimension, my_grid->dimension * sizeof(float));
	cudaBindTexture2D(NULL, outputTex1D, grid2_d, desc, my_grid->dimension, my_grid->dimension, my_grid->dimension * sizeof(float));
	cudaBindTexture2D(NULL, diffTex1D, diff_d, desc, my_grid->dimension, my_grid->dimension, my_grid->dimension * sizeof(float));

	
	printf("Solving using GPU Texture Memory. \n");
	
	done = 0;
	num_iter = 0;
	diff = 0.0f;
	time = 0.0f;
	
	while(!done)
	{
		gettimeofday(&start, NULL);
		solver_kernel_optimized<<< dimGrid, dimBlock >>>(grid1_d,grid2_d, GRID_DIMENSION, diff_d);
		cudaThreadSynchronize();

		gettimeofday(&stop, NULL);
		
		time += (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000);

		float* temp = grid1_d;
		grid1_d = grid2_d;
		grid2_d = temp;
		cudaMemcpy(diff_h, diff_d, sizeof(float)*my_grid->num_elements, cudaMemcpyDeviceToHost);
		gettimeofday(&start, NULL);
		for(int i = 0; i < my_grid->num_elements; i++)
			diff += diff_h[i];
		if(diff/(my_grid->num_elements) < (float) TOLERANCE)
			done = 1;
		gettimeofday(&stop, NULL);
		time += (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000);

		//printf("Difference: %f, Iteration: %d\n", diff/(my_grid->num_elements), num_iter);

		diff=0;
		num_iter++;
	}
	
	printf("Execution time = %fs. \n", time);
	
	check_for_error("KERNEL FAILURE");

	cudaStatus = cudaMemcpy(my_grid->element, grid1_d, sizeof(float) * my_grid->num_elements, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		const char *str = (char*) malloc(1024); // To store error string
		str = cudaGetErrorString(cudaStatus);
		fprintf(stderr, "CUDA Error!:: %s\n", str);
		getchar();
	}
	
	
	cudaFree(grid1_d);
	cudaFree(grid2_d);
	cudaFree(diff_d);
	
	/*cudaUnbindTexture(inputTex1D);
	cudaUnbindTexture(outputTex1D);
	cudaUnbindTexture(diffTex1D); */

	
	free(grid2_h);
}

/* The main function */
int 
main(int argc, char **argv)
{	
	/* Generate the grid */
	GRID_STRUCT *grid_for_cpu = (GRID_STRUCT *)malloc(sizeof(GRID_STRUCT)); // The grid data structure
	GRID_STRUCT *grid_for_gpu = (GRID_STRUCT *)malloc(sizeof(GRID_STRUCT)); // The grid data structure

	grid_for_cpu->dimension = GRID_DIMENSION;
	grid_for_cpu->num_elements = grid_for_cpu->dimension * grid_for_cpu->dimension;
	grid_for_gpu->dimension = GRID_DIMENSION;
	grid_for_gpu->num_elements = grid_for_gpu->dimension * grid_for_gpu->dimension;

 	create_grids(grid_for_cpu, grid_for_gpu); // Create the grids and populate them with the same set of random values
	
	struct timeval start, stop;	
	gettimeofday(&start, NULL);
	
	printf("Using the cpu to solve the grid. \n");
	compute_gold(grid_for_cpu);  // Use CPU to solve 
	
	gettimeofday(&stop, NULL);
	printf("Execution time = %fs. \n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));

	// Use the GPU to solve the equation
	compute_on_device(grid_for_gpu);
	
	// Print key statistics for the converged values
	printf("CPU: \n");
	print_statistics(grid_for_cpu);

	printf("GPU: \n");
	print_statistics(grid_for_gpu);
	
    /* Compute grid differences. */
    compute_grid_differences(grid_for_cpu, grid_for_gpu);

	free((void *)grid_for_cpu->element);	
	free((void *)grid_for_cpu); // Free the grid data structure 
	
	free((void *)grid_for_gpu->element);	
	free((void *)grid_for_gpu); // Free the grid data structure 

	exit(0);
}

// This function checks for errors returned by the CUDA run time
void check_for_error(char *msg){
	cudaError_t err = cudaGetLastError();
	if(cudaSuccess != err){
		printf("CUDA ERROR: %s (%s). \n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
} 
