/* Author: Trung Do
 * 
 *  
 *   * Compile as follows:
 *    * gcc -o solver solver.c solver_gold.c -std=c99 -lm -lpthread 
 *     */

#include <stdio.h>
#include <string.h>
#include <malloc.h>
#include <time.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>
#include "grid.h" 

#define NUM_THREADS 8

extern int compute_gold(GRID_STRUCT *);
int compute_using_pthreads_jacobi(GRID_STRUCT *);
int compute_using_pthreads_red_black(GRID_STRUCT *);
void compute_grid_differences(GRID_STRUCT *, GRID_STRUCT *, GRID_STRUCT *);
void* jacobi (void *args);

typedef struct args_for_thread{
	int thread_id;
	int num_elements;
	int chunk_size;
	int offset;
	int endpoint;
	GRID_STRUCT *my_grid;
	GRID_STRUCT *temp;
	float diff;
	int done;
	int num_iter;
} ARGS_FOR_THREAD;

typedef struct barrier_struct{
	pthread_mutex_t mutex;
	pthread_cond_t condition;
	int counter;
} BARRIER;

void barrier_sync(BARRIER* barrier);

pthread_mutex_t mutex;
BARRIER barrier = {PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, 0};

/* This function prints the grid on the screen. */
void 
display_grid(GRID_STRUCT *my_grid)
{
    for(int i = 0; i < my_grid->dimension; i++)
        for(int j = 0; j < my_grid->dimension; j++)
            printf("%f \t", my_grid->element[i * my_grid->dimension + j]);
   		
    printf("\n");
}


/* Print out statistics for the converged values, including min, max, and average. */
void 
print_statistics(GRID_STRUCT *my_grid)
{
    float min = INFINITY;
    float max = 0.0;
    double sum = 0.0; 
    
    for(int i = 0; i < my_grid->dimension; i++){
        for(int j = 0; j < my_grid->dimension; j++){
            sum += my_grid->element[i * my_grid->dimension + j];
           
            if(my_grid->element[i * my_grid->dimension + j] > max) 
                max = my_grid->element[i * my_grid->dimension + j];

				if(my_grid->element[i * my_grid->dimension + j] < min) 
                    min = my_grid->element[i * my_grid->dimension + j];
				 
        }
    }

    printf("AVG: %f \n", sum/(float)my_grid->num_elements);
	printf("MIN: %f \n", min);
	printf("MAX: %f \n", max);

	printf("\n");
}

/* Calculate the differences between grid elements for the various implementations. */
void compute_grid_differences(GRID_STRUCT *grid_1, GRID_STRUCT *grid_2, GRID_STRUCT *grid_3)
{
    float diff_12, diff_13;
    int dimension = grid_1->dimension;
    int num_elements = dimension*dimension;

    diff_12 = 0.0;
    diff_13 = 0.0;
    for(int i = 0; i < grid_1->dimension; i++){
        for(int j = 0; j < grid_1->dimension; j++){
            diff_12 += fabsf(grid_1->element[i * dimension + j] - grid_2->element[i * dimension + j]);
            diff_13 += fabsf(grid_1->element[i * dimension + j] - grid_3->element[i * dimension + j]);
        }
    }
    printf("Average difference in grid elements for Gauss Seidel and Red-Black methods = %f. \n", \
            diff_12/num_elements);

    printf("Average difference in grid elements for Gauss Seidel and Jacobi methods = %f. \n", \
            diff_13/num_elements);


}

/* Create a grid of random floating point values bounded by UPPER_BOUND_ON_GRID_VALUE */
void 
create_grids(GRID_STRUCT *grid_1, GRID_STRUCT *grid_2, GRID_STRUCT *grid_3)
{
	printf("Creating a grid of dimension %d x %d. \n", grid_1->dimension, grid_1->dimension);
	grid_1->element = (float *)malloc(sizeof(float) * grid_1->num_elements);
	grid_2->element = (float *)malloc(sizeof(float) * grid_2->num_elements);
	grid_3->element = (float *)malloc(sizeof(float) * grid_3->num_elements);

	srand((unsigned)time(NULL));
	
	float val;
	for(int i = 0; i < grid_1->dimension; i++)
		for(int j = 0; j < grid_1->dimension; j++){
			val =  ((float)rand()/(float)RAND_MAX) * UPPER_BOUND_ON_GRID_VALUE;
			grid_1->element[i * grid_1->dimension + j] = val; 	
			grid_2->element[i * grid_2->dimension + j] = val; 
			grid_3->element[i * grid_3->dimension + j] = val; 
			
		}
}

/* Edit this function to use the jacobi method of solving the equation. The final result should 
 *  * be placed in the grid_2 data structure */

int compute_using_pthreads_jacobi(GRID_STRUCT *grid_2)
{
		  pthread_t thread_id[NUM_THREADS]; // Data structure to store the thread IDs
		  pthread_attr_t attributes; // Thread attributes
		  pthread_attr_init(&attributes); // Initialize the thread attributes to the default values

		  ARGS_FOR_THREAD *args_for_thread;
	
		  pthread_mutex_t mutex_for_diff; // Lock for the shared variable diff
		  pthread_mutex_init(&mutex_for_diff, NULL); // Initialize the mutex
		  //int k = grid_2->num_elements;
		  	    GRID_STRUCT *temp = (GRID_STRUCT *)malloc(sizeof(GRID_STRUCT));// Generate a grid for store even iteration
		  	    temp->dimension = grid_2->dimension;
		  	    temp->num_elements = grid_2->num_elements;
		  	    temp->element = (float *)malloc(sizeof(float) * grid_2->num_elements);
		  	    for(int m = 0; m < temp->dimension; m++){ //Initilize temp to grid_2
					for(int n = 0; n < temp->dimension; n++){
		  	      		temp->element[m * grid_2->dimension + n] = grid_2->element[m * grid_2->dimension + n];
		  	      	}
		  	    }
		    /* Allocate memory on the heap for the required data structures and create the worker threads. */
		 	int num_iter = 0;
		  	int i;
		  	float diff = 0;
			int done = 0;
			//args_for_thread[NUM_THREADS];
			//Create the barrier data structure and initialize it

			for(i = 0; i < NUM_THREADS; i++){
				args_for_thread = (ARGS_FOR_THREAD *)malloc(sizeof(ARGS_FOR_THREAD));
				args_for_thread->thread_id = i; // Provide thread ID
				args_for_thread->num_elements = grid_2->num_elements;
				args_for_thread->chunk_size = (grid_2->dimension-2)/ NUM_THREADS; // Chunk size
				args_for_thread->offset = (i * args_for_thread->chunk_size + 1); // Starting offset
				args_for_thread->endpoint = ((i+1)*args_for_thread->chunk_size); //Ending point for no-end thread
				args_for_thread->my_grid = grid_2;
				args_for_thread->temp = temp;
				args_for_thread->diff = diff;
				args_for_thread->done = done;
				args_for_thread->num_iter = num_iter;
			
				if(pthread_create(&thread_id[i], NULL ,jacobi,(void *)args_for_thread) !=0 ){
					printf("Cannot create thread\n");
					exit(0);
				}
				else
					printf("Thread %d created.\n", i);
			}	
			/* Wait for the workers to finish. */
			for(i = 0; i < NUM_THREADS; i++)
				 pthread_join(thread_id[i], NULL);
				 // Free args_for_thread structures
			for(i = 0; i < NUM_THREADS; i++)
				free((void *)args_for_thread);
			return num_iter;
		}

void *jacobi (void *args)
{
	/* Typecast the argument to a pointer to the ARGS_FOR_THREAD structure. */
	ARGS_FOR_THREAD *args_for_me = (ARGS_FOR_THREAD *)args; 		  // print_args(args_for_me)
	float partial_diff = 0;
	int dimension = args_for_me->my_grid->dimension;

	if(args_for_me->thread_id < (NUM_THREADS-1)){ // this code for all thread except the last one
		while(!args_for_me->done){ // while not convergen yet
		    partial_diff = 0;
			if (((args_for_me->num_iter)% 2) == 0 ) {// Even number of iteration
				for(int i = args_for_me->offset; i <= args_for_me->endpoint; i++){
					for(int j = 1; j < (dimension-1); j++){
						// apply Jacobi update rule
						args_for_me->temp->element[i * dimension + j] = 0.20*(args_for_me->my_grid->element[i * dimension + j] +
							args_for_me->my_grid->element[(i - 1) * dimension + j] +
								args_for_me->my_grid->element[(i + 1) * dimension + j] +
									args_for_me->my_grid->element[i * dimension + (j + 1)] +
									args_for_me->my_grid->element[i * dimension + (j - 1)]);
							partial_diff = partial_diff + fabs(args_for_me->my_grid->element[i * dimension + j] - args_for_me->temp->element[i * dimension + j]);
					}
				}
			}

			else{ // The odd iteration we ping pong between two matrixs
				for(int i = args_for_me->offset; i<= args_for_me->endpoint; i++){
					for(int j = 1; j < (dimension - 1); j++){
						//Apply jacobi rule
						//
						printf("test: %d\n",args_for_me->my_grid->element[i * dimension + j]);
						int temp = 0;
						/*args_for_me->my_grid->element[i * dimension + j]*/temp = 0.20*(args_for_me->temp->element[i * dimension + j] + \
						args_for_me->temp->element[(i - 1) * dimension + j] + \
						args_for_me->temp->element[(i + 1) * dimension + j] + \
						args_for_me->temp->element[i * dimension + (j + 1)] + \
						args_for_me->temp->element[i * dimension + (j - 1)]);
						
						printf("Fabs: %d\n", fabs(args_for_me->temp->element[i * dimension + j]));
						printf("my_grid: %d\n", /*args_for_me->my_grid->element[i * dimension + j]*/temp);
						
						partial_diff += fabs(args_for_me->temp->element[i * dimension + j] - args_for_me->my_grid->element[i * dimension + j]);
					}
				}
			}

			/* Accumulate partial sums into the shared variable. */
			//pthread_mutex_lock(&mutex);
			args_for_me->diff += partial_diff;
			//pthread_mutex_unlock(&mutex);
			/* Barrier thread synchonization */
			barrier_sync(&barrier);
			printf("Partial diff: %d\n",partial_diff);
			printf("Iteration %d completed. Diff is: %d\n",args_for_me->num_iter, args_for_me->diff);
			args_for_me->num_iter++;

			if((float)args_for_me->diff/((float)(args_for_me->my_grid->dimension*args_for_me->my_grid->dimension)) < (float)TOLERANCE) {
           			printf("Done\n"); 
				args_for_me->done = 1;
			}
		}
	} 
	else{ //this code use for the last thread that compute the rest elements
		while(!args_for_me->done){ // Start an iteration
			 partial_diff = 0;
			 if ((args_for_me->num_iter % 2) == 0 ) { // Even number of iteration
			for(int i = args_for_me->offset; i < (dimension-1); i++){
				for(int j = 1; j < (dimension-1); j++){
					//Apply Jacobi update rule
					args_for_me->temp->element[i * dimension + j] = 0.20*(args_for_me->my_grid->element[i * dimension + j] + \
					args_for_me->my_grid->element[(i - 1) * dimension + j] + \
					args_for_me->my_grid->element[(i + 1) * dimension + j] + \
					args_for_me->my_grid->element[i * dimension + (j + 1)] + \
					args_for_me->my_grid->element[i * dimension + (j - 1)]);
					partial_diff = partial_diff + fabs(args_for_me->my_grid->element[i * dimension + j] - args_for_me->temp->element[i * dimension + j]);
				}
			}
		}

		else{ // Odd number of iteration
		for(int i = args_for_me->offset; i< (dimension -1); i++){
			for(int j = 1; j < (dimension - 1); j++){
				args_for_me->my_grid->element[i * dimension + j] = 0.20*(args_for_me->temp->element[i * dimension + j] + \
				args_for_me->temp->element[(i - 1) * dimension + j] + \
				args_for_me->temp->element[(i + 1) * dimension + j] + \
				args_for_me->temp->element[i * dimension + (j + 1)] + \
				args_for_me->temp->element[i * dimension + (j - 1)]);

				partial_diff = partial_diff + fabs(args_for_me->temp->element[i * dimension + j] - args_for_me->my_grid->element[i * dimension + j]);
			}
		}	
	}
		/* Accumulate partial diff into the shared variable. */
		pthread_mutex_lock(&mutex);
		args_for_me->diff += partial_diff;
		pthread_mutex_unlock(&mutex);

		/* Barrier thread synchonization */
		barrier_sync(&barrier);

	if((float)args_for_me->diff/((float)(args_for_me->my_grid->dimension*args_for_me->my_grid->dimension)) < (float)TOLERANCE) {
            		args_for_me->done = 1;
			printf("Done!");
	}
		}
	}
return NULL;
}

/* Edit this function to use the red-black method of solving the equation. The final result 
 *  * should be placed in the grid_3 data structure */
int 
compute_using_pthreads_red_black(GRID_STRUCT *grid_3)
{		
}

		
/* The main function */
int 
main(int argc, char **argv)
{	
	/* Generate the grids and populate them with the same set of random values. */
	GRID_STRUCT *grid_1 = (GRID_STRUCT *)malloc(sizeof(GRID_STRUCT)); 
	GRID_STRUCT *grid_2 = (GRID_STRUCT *)malloc(sizeof(GRID_STRUCT)); 
	GRID_STRUCT *grid_3 = (GRID_STRUCT *)malloc(sizeof(GRID_STRUCT)); 

	grid_1->dimension = GRID_DIMENSION;
	grid_1->num_elements = grid_1->dimension * grid_1->dimension;
	grid_2->dimension = GRID_DIMENSION;
	grid_2->num_elements = grid_2->dimension * grid_2->dimension;
	grid_3->dimension = GRID_DIMENSION;
	grid_3->num_elements = grid_3->dimension * grid_3->dimension;

 	create_grids(grid_1, grid_2, grid_3);

	
	/* Compute the reference solution using the single-threaded version. */
	//printf("Using the single threaded version to solve the grid. \n");
	int num_iter/* = compute_gold(grid_1)*/;
	printf("Convergence achieved after %d iterations. \n", num_iter);

	/* Use pthreads to solve the equation uisng the red-black parallelization technique. */
	printf("Using pthreads to solve the grid using the red-black parallelization method. \n");
	num_iter = compute_using_pthreads_red_black(grid_2);
	printf("Convergence achieved after %d iterations. \n", num_iter);

	
	/* Use pthreads to solve the equation using the jacobi method in parallel. */
	printf("Using pthreads to solve the grid using the jacobi method. \n");
	num_iter = compute_using_pthreads_jacobi(grid_3);
	printf("Convergence achieved after %d iterations. \n", num_iter);

	/* Print key statistics for the converged values. */
	printf("\n");
	printf("Reference: \n");
	print_statistics(grid_1);

	printf("Red-black: \n");
	print_statistics(grid_2);
		
	printf("Jacobi: \n");
	print_statistics(grid_3);

    /* Compute grid differences. */
    compute_grid_differences(grid_1, grid_2, grid_3);

	/* Free up the grid data structures. */
	free((void *)grid_1->element);	
	free((void *)grid_1); 
	
	free((void *)grid_2->element);	
	free((void *)grid_2);

	free((void *)grid_3->element);	
	free((void *)grid_3);

	return 0;
}

void barrier_sync(BARRIER* barrier)
{
	pthread_mutex_lock(&(barrier->mutex));
	barrier->counter++;
	
	if(barrier->counter == NUM_THREADS)
	{
		barrier->counter = 0;
		pthread_cond_broadcast(&(barrier->condition)); 
	}
	else
	{
		while((pthread_cond_wait(&(barrier->condition), &(barrier->mutex))) != 0);
	}

	pthread_mutex_unlock(&(barrier->mutex));
}

