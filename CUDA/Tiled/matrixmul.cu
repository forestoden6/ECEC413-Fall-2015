/* Matrix multiplication: C = A * B.
 * Host code.

 * Author: Naga Kandasamy
 * Date modified: 11/05/2015
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

/* Inlcude the kernel code here. */
#include "matrixmul_kernel.cu"


extern "C" void computeGold(float*, const float*, const float*, unsigned int, unsigned int, unsigned int);
Matrix AllocateDeviceMatrix(const Matrix M);
Matrix AllocateMatrix(int height, int width, int init);
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost);
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice);
void FreeDeviceMatrix(Matrix* M);
void FreeMatrix(Matrix* M);
void MatrixMulOnDevice(const Matrix M, const Matrix N, Matrix P);
void checkCUDAError(const char *msg);
int checkResults(float *, float *, int, float);


int 
main(int argc, char** argv) {

	Matrix  M, N, P;
	
	srand(52);
	
	// Allocate and initialize the matrices
	M  = AllocateMatrix(MATRIX_SIZE, MATRIX_SIZE, 1);
	N  = AllocateMatrix(MATRIX_SIZE, MATRIX_SIZE, 1);
	P  = AllocateMatrix(MATRIX_SIZE, MATRIX_SIZE, 0);

	// M * N on the device
	MatrixMulOnDevice(M, N, P);
	printf("GPU computation complete\n");


    printf("Multiplying matrices on the CPU. \n");
    struct timeval start, stop;	
	gettimeofday(&start, NULL);

    Matrix reference = AllocateMatrix(P.height, P.width, 0);    /* Compute M * N on the CPU. */
	computeGold(reference.elements, M.elements, N.elements, M.height, M.width, N.width);

    gettimeofday(&stop, NULL);
	printf("Execution time = %fs. \n", (float)(stop.tv_sec - start.tv_sec +\\
                (stop.tv_usec - start.tv_usec)/(float)1000000));


    /* Check if the device result is equivalent to the expected solution. */
    int num_elements = M.height*M.width;
	int status = checkResults(reference.elements, P.elements, num_elements, 0.001f);
	printf("Test %s\n", (1 == status) ? "PASSED" : "FAILED");

	// Free matrices
    FreeMatrix(&M);
    FreeMatrix(&N);
    FreeMatrix(&P);
	
    return 0;
}


void 
MatrixMulOnDevice(const Matrix M, const Matrix N, Matrix P)
{
    // Load M and N to the device
    Matrix Md = AllocateDeviceMatrix(M);
    CopyToDeviceMatrix(Md, M);
    Matrix Nd = AllocateDeviceMatrix(N);
    CopyToDeviceMatrix(Nd, N);

    // Allocate P on the device
    Matrix Pd = AllocateDeviceMatrix(P);
    CopyToDeviceMatrix(Pd, P); // Clear memory

    // Setup the execution configuration
    dim3 dimBlock, dimGrid;
    dimBlock.x = dimBlock.y = TILE_SIZE;
    dimBlock.z = 1;
    dimGrid.x = (P.width / dimBlock.x) + ((P.width % dimBlock.x) ? 1:0 );
    dimGrid.y = (P.height / dimBlock.y) + ((P.height % dimBlock.y) ? 1:0 );
    dimGrid.z = 1;

    printf("Setting up a %d x %d grid of thread blocks. \n", dimGrid.x, dimGrid.y);

    struct timeval start, stop;	
	gettimeofday(&start, NULL);

	// Launch the device computation threads!
	MatrixMulKernel<<<dimGrid,dimBlock>>>(Md,Nd,Pd); 
	cudaThreadSynchronize();

    gettimeofday(&stop, NULL);
	printf("Execution time = %fs. \n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));

    checkCUDAError("Error in kernel");  

	
    // Read P from the device
    CopyFromDeviceMatrix(P, Pd); 

    // Free device matrices
    FreeDeviceMatrix(&Md);
    FreeDeviceMatrix(&Nd);
    FreeDeviceMatrix(&Pd);
}

// Allocate a device matrix of same size as M.
Matrix 
AllocateDeviceMatrix(const Matrix M)
{
    Matrix Mdevice = M;
    int size = M.width * M.height * sizeof(float);
    cudaMalloc((void**)&Mdevice.elements, size);
    return Mdevice;
}

// Allocate a device matrix of dimensions height*width
//	If init == 0, initialize to all zeroes.  
//	If init == 1, perform random initialization.
//  If init == 2, initialize matrix parameters, but do not allocate memory 
Matrix 
AllocateMatrix(int height, int width, int init)
{
    Matrix M;
    M.width = M.pitch = width;
    M.height = height;
    int size = M.width * M.height;
    M.elements = NULL;
    
    // don't allocate memory on option 2
    if(init == 2)
		return M;
		
	M.elements = (float*) malloc(size*sizeof(float));

	for(unsigned int i = 0; i < M.height * M.width; i++){
		M.elements[i] = (init == 0) ? (0.0f) : ((rand()*3 / (float)RAND_MAX));
	}

    return M;
}	

// Copy a host matrix to a device matrix.
void 
CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost)
{
    int size = Mhost.width * Mhost.height * sizeof(float);
    Mdevice.height = Mhost.height;
    Mdevice.width = Mhost.width;
    Mdevice.pitch = Mhost.pitch;
    cudaMemcpy(Mdevice.elements, Mhost.elements, size, cudaMemcpyHostToDevice);
}

// Copy a device matrix to a host matrix.
void 
CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice)
{
    int size = Mdevice.width * Mdevice.height * sizeof(float);
    cudaMemcpy(Mhost.elements, Mdevice.elements, size, cudaMemcpyDeviceToHost);
}

// Free a device matrix.
void 
FreeDeviceMatrix(Matrix* M)
{
    cudaFree(M->elements);
    M->elements = NULL;
}

// Free a host Matrix
void 
FreeMatrix(Matrix* M)
{
    free(M->elements);
    M->elements = NULL;
}


void 
checkCUDAError(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if( cudaSuccess != err) 
	{
		printf("CUDA ERROR: %s (%s).\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}						 
}


int 
checkResults(float *reference, float *gpu_result, int num_elements, float threshold)
{
    int checkMark = 1;
    float epsilon = 0.0;
    
    for(int i = 0; i < num_elements; i++)
        if(fabsf((reference[i] - gpu_result[i])/reference[i]) > threshold){
            checkMark = 0;
        }

    for(int i = 0; i < num_elements; i++)
        if(fabsf((reference[i] - gpu_result[i])/reference[i]) > epsilon){
            epsilon = fabsf((reference[i] - gpu_result[i])/reference[i]);
        }

    printf("Max epsilon = %f. \n", epsilon); 
    return checkMark;
}
