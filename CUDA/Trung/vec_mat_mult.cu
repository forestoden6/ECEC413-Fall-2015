/* Vector-matrix multiplication: Y = A * X.
 * Host code.
 * Author: Trung Do
 * Date: 11/15/2015
*/

#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>

// includes, project
//#include <helper_cuda.h>

// includes, kernels
#include "vec_mat_mult_kernel.cu"

#define MIN_NUMBER 1
#define MAX_NUMBER 4

////////////////////////////////////////////////////////////////////////////////
// declarations, forward

extern "C"
void compute_gold(float*, const float*, const float*, unsigned int, unsigned int);

Matrix allocate_matrix_on_gpu(const Matrix M);
Matrix allocate_matrix(int num_rows, int num_columns, int init);
void copy_matrix_to_device(Matrix Mdevice, const Matrix Mhost);
void copy_matrix_from_device(Matrix Mhost, const Matrix Mdevice);
void vec_mat_mult_on_device_using_global_memory(const Matrix M, const Matrix N, Matrix P);
void vec_mat_mult_on_device_using_shared_memory(const Matrix M, const Matrix N, Matrix P);
void print_matrix(const Matrix M);
float get_random_number(int, int);
int checkResults(float *, float *, int, float);


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int 
main(int argc, char** argv) {
	// Matrices for the program
	Matrix  A; // N x N matrix
	Matrix  X; // N x 1 vector
	Matrix  Y_cpu, Y_gpu_1, Y_gpu_2; // N x 1 vector
	
	// Initialize the random number generator with a seed value 
	srand(time(NULL));
	
	// Check command line arguments
	if(argc > 1){
		printf("Error. This program accepts no arguments. \n");
		exit(0);
	}		
	 
	// Allocate and initialize the matrices
	A  = allocate_matrix(MATRIX_SIZE, MATRIX_SIZE, 1); // Create a random N x N matrix
	X  = allocate_matrix(MATRIX_SIZE, 1, 1); // Create a random N x 1 vector 
	Y_cpu  = allocate_matrix(MATRIX_SIZE, 1, 0); // Allocate memory for the output vectors
	Y_gpu_1 = allocate_matrix(MATRIX_SIZE, 1, 0); 
	Y_gpu_2 = allocate_matrix(MATRIX_SIZE, 1, 0);
 
    	// compute the vector-matrix multiplication on the CPU for comparison    

	struct timeval start, stop;	
	gettimeofday(&start, NULL);	
	printf("Performing serial calculation using CPU. \n");
	
	compute_gold(Y_cpu.elements, A.elements, X.elements, A.num_rows, A.num_columns);
	
	gettimeofday(&stop, NULL);
	printf("Execution time = %fs. \n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));

	
	// Perform the vector-matrix multiplication on the GPU using global memory
    	// Return the results in Y_gpu_1
	vec_mat_mult_on_device_using_global_memory(A, X, Y_gpu_1);
   
	// check if the device result is equivalent to the expected solution
    	printf("Checking against reference result. \n");
	int size_elements = NUM_ROWS;
	int res = checkResults(Y_cpu.elements, Y_gpu_1.elements, size_elements, 0.0001);
	printf("Test %s\n", (1 == res) ? "PASSED" : "FAILED");


   	 // Perform the vector-matrix multiplication on the GPU using global memory
    	// Return the results in Y_gpu_2
	vec_mat_mult_on_device_using_shared_memory(A, X, Y_gpu_2);
   
	// check if the device result is equivalent to the expected solution
   	 printf("Checking against reference result. \n");
    	res = checkResults(Y_cpu.elements, Y_gpu_2.elements, size_elements, 0.0001);
	printf("Test %s\n", (1 == res) ? "PASSED" : "FAILED");

	// Free host matrices
	free(A.elements); A.elements = NULL;
	free(X.elements); X.elements = NULL;
	free(Y_cpu.elements); Y_cpu.elements = NULL;
	free(Y_gpu_1.elements); Y_gpu_1.elements = NULL;
    free(Y_gpu_2.elements); Y_gpu_2.elements = NULL;

	return 0;
}


////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void vec_mat_mult_on_device_using_global_memory(const Matrix A, const Matrix X, Matrix Y){
	//Interface host call to the device kernel code and invoke the kernel	
	
	/* allocate space on gpu */
	Matrix gpu_a = allocate_matrix_on_gpu( A );
	Matrix gpu_x = allocate_matrix_on_gpu( X );
	Matrix gpu_y = allocate_matrix_on_gpu( Y );

	/* copy matrices to gpu  */
	copy_matrix_to_device( gpu_a, A );
	copy_matrix_to_device( gpu_x, X );

	/* set gpu parameters */
	dim3 dimBlock( 512, 1 );
	dim3 dimGrid( MATRIX_SIZE / dimBlock.x, 1 );

	struct timeval start, stop;	
	gettimeofday(&start, NULL);
	
	/* invoke kernel on gpu*/
	printf("Performing multiplication using global memory. \n");
	MatrixMulKernel_using_global_memory <<< dimGrid, dimBlock >>> (gpu_a.elements, gpu_x.elements, gpu_y.elements);

	/* end timer, print result */
	cudaThreadSynchronize();
	gettimeofday(&stop, NULL);
	printf("Execution time = %fs. \n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));

	/* check for gpu errors */
	cudaError_t err =  cudaGetLastError();
	if ( cudaSuccess != err ) {
		fprintf(stderr, " GPU kernel failed: %s.\n", cudaGetErrorString(err));
	}

	/* copy result back to host */
	copy_matrix_from_device( Y, gpu_y );

	/* clean up mem on gpu */
	cudaFree( gpu_a.elements );
	cudaFree( gpu_x.elements );
	cudaFree( gpu_y.elements );
}

void vec_mat_mult_on_device_using_shared_memory(const Matrix A, const Matrix X, Matrix Y){
	//Interface host call to the device kernel code and invoke the kernel	

	/* allocate space on gpu */
        Matrix gpu_a = allocate_matrix_on_gpu( A );
        Matrix gpu_x = allocate_matrix_on_gpu( X );
        Matrix gpu_y = allocate_matrix_on_gpu( Y );

        /* copy matrices to gpu  */
        copy_matrix_to_device( gpu_a, A );
        copy_matrix_to_device( gpu_x, X );

        /* set gpu parameters */
        dim3 dimBlock( 16, 16 );
        dim3 dimGrid( MATRIX_SIZE / dimBlock.x, 1 );

			struct timeval start, stop;	
	gettimeofday(&start, NULL);

		printf("Performing multiplication using shared memory. \n");
        /* invoke kernel on gpu*/
        MatrixMulKernel_using_shared_memory <<< dimGrid, dimBlock >>> (gpu_a.elements, gpu_x.elements, gpu_y.elements);

	/* end timer, print result */
	cudaThreadSynchronize();
	gettimeofday(&stop, NULL);
	printf("Execution time = %fs. \n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));
        
	/* check for gpu errors */
        cudaError_t err =  cudaGetLastError();
        if ( cudaSuccess != err ) {
                fprintf(stderr, " GPU kernel failed: %s.\n", cudaGetErrorString(err));
        } 

        /* copy result back to host */
        copy_matrix_from_device( Y, gpu_y );

        /* clean up mem on gpu */
        cudaFree( gpu_a.elements );
        cudaFree( gpu_x.elements );
        cudaFree( gpu_y.elements );
}


// Allocate a device matrix of same size as M.
Matrix allocate_matrix_on_gpu(const Matrix M){
    Matrix Mdevice = M;
    int size = M.num_rows * M.num_columns * sizeof(float);
    cudaMalloc((void**)&Mdevice.elements, size);
    return Mdevice;
}

// Allocate a matrix of dimensions height*width
//	If init == 0, initialize to all zeroes.  
//	If init == 1, perform random initialization.
Matrix allocate_matrix(int num_rows, int num_columns, int init){
    	Matrix M;
    	M.num_columns = M.pitch = num_columns;
    	M.num_rows = num_rows;
    	int size = M.num_rows * M.num_columns;
		
	M.elements = (float*) malloc(size*sizeof(float));
	for(unsigned int i = 0; i < size; i++){
		if(init == 0) M.elements[i] = 0; 
		else
			M.elements[i] = get_random_number(MIN_NUMBER, MAX_NUMBER);
	}
    return M;
}	

// Copy a host matrix to a device matrix.
void copy_matrix_to_device(Matrix Mdevice, const Matrix Mhost)
{
    int size = Mhost.num_rows * Mhost.num_columns * sizeof(float);
    Mdevice.num_rows = Mhost.num_rows;
    Mdevice.num_columns = Mhost.num_columns;
    Mdevice.pitch = Mhost.pitch;
    cudaMemcpy(Mdevice.elements, Mhost.elements, size, cudaMemcpyHostToDevice);
}

// Copy a device matrix to a host matrix.
void copy_matrix_from_device(Matrix Mhost, const Matrix Mdevice){
    int size = Mdevice.num_rows * Mdevice.num_columns * sizeof(float);
    cudaMemcpy(Mhost.elements, Mdevice.elements, size, cudaMemcpyDeviceToHost);
}

// Prints the matrix out to screen
void print_matrix(const Matrix M){
	for(unsigned int i = 0; i < M.num_rows; i++){
		for(unsigned int j = 0; j < M.num_columns; j++)
			printf("%f ", M.elements[i*M.num_columns + j]);
		printf("\n");
	} 
	printf("\n");
}

// Returns a random floating-point number between the specified min and max values 
float get_random_number(int min, int max){
	return (float)floor((double)(min + (max - min + 1)*((float)rand()/(float)RAND_MAX)));
}

int 
checkResults(float *reference, float *gpu_result, int num_elements, float threshold)
{
    int checkMark = 1;
    float epsilon = 0.0;
    
    for(int i = 0; i < num_elements; i++)
        if(fabsf((reference[i] - gpu_result[i])/reference[i]) > threshold){
            checkMark = 0;
            break;
        }

    for(int i = 0; i < num_elements; i++)
        if(fabsf((reference[i] - gpu_result[i])/reference[i]) > epsilon){
            epsilon = fabsf((reference[i] - gpu_result[i])/reference[i]);
        }

    printf("Max epsilon = %f. \n", epsilon); 
    return checkMark;
}

