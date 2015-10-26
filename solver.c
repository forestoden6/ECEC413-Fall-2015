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

#define NUM_THREADS 4

extern int compute_gold(GRID_STRUCT *);
int compute_using_pthreads_jacobi(GRID_STRUCT *);
int compute_using_pthreads_red_black(GRID_STRUCT *);
void compute_grid_differences(GRID_STRUCT *, GRID_STRUCT *, GRID_STRUCT *);
void* jacobi (void *args);
void* red_black(void *args);

typedef struct args_for_thread{
	int thread_id;
	int num_elements;
	GRID_STRUCT *my_grid;
	GRID_STRUCT *temp;
} ARGS_FOR_THREAD;

typedef struct barrier_struct{
	pthread_mutex_t mutex;
	pthread_cond_t condition;
	int counter;
} BARRIER;

float diff = 0;
int is_red = 1;	
int num_iter = 0;
int done = 0;


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
			temp->element = (float *)malloc(sizeof(float) * temp->num_elements);
			for(int m = 0; m < temp->dimension; m++){ //Initilize temp to grid_2
				for(int n = 0; n < temp->dimension; n++){
					temp->element[m * grid_2->dimension + n] = grid_2->element[m * grid_2->dimension + n];
				}
			}
		    /* Allocate memory on the heap for the required data structures and create the worker threads. */
		  	int i;
			int done = 0;
			//args_for_thread[NUM_THREADS];
			//Create the barrier data structure and initialize it

			for(i = 0; i < NUM_THREADS; i++){
				args_for_thread = (ARGS_FOR_THREAD *)malloc(sizeof(ARGS_FOR_THREAD));
				args_for_thread->thread_id = i; // Provide thread ID
				args_for_thread->num_elements = grid_2->num_elements;
				args_for_thread->my_grid = grid_2;
				args_for_thread->temp = temp;
				
				printf("Args for thread: %d\n", grid_2->element[0]);
			
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
			free((void *)args_for_thread);
			return num_iter;
		}   

void * jacobi(void *args){
	ARGS_FOR_THREAD *args_for_me = (ARGS_FOR_THREAD *)args;
	//Reset global variables after red-black 
	done = 0;
	diff = 0;
	num_iter = 0;
	
	float local_diff = 0;
	
	/* While not converged */
	while(done != 1){
		local_diff = 0;
		if(num_iter % 2 == 0){ /* Every even iteration move from my_grid to temp grid  */
			for(int i = args_for_me->thread_id + 1; i < args_for_me->my_grid->dimension-1; i+=NUM_THREADS){ /* Row */
				for(int j = 1; j < args_for_me->my_grid->dimension-1; j++){ /* Col */
					int pos = i * args_for_me->my_grid->dimension + j;
					args_for_me->temp->element[pos] = \
						0.20*(args_for_me->my_grid->element[pos] + \
						   args_for_me->my_grid->element[(i - 1) * args_for_me->my_grid->dimension + j] +\
						   args_for_me->my_grid->element[(i + 1) * args_for_me->my_grid->dimension + j] +\
						   args_for_me->my_grid->element[i * args_for_me->my_grid->dimension + (j + 1)] +\
						   args_for_me->my_grid->element[i * args_for_me->my_grid->dimension + (j - 1)]);
				local_diff += fabs(args_for_me->my_grid->element[pos] - args_for_me->temp->element[pos]);
				/* Store difference in local variable */
				}
			}
		}
		else{ /* Move from temp to my_grid */
			for(int i = args_for_me->thread_id + 1; i < args_for_me->my_grid->dimension-1; i+=NUM_THREADS){
				for(int j = 1; j < args_for_me->my_grid->dimension-1; j++){
					int pos = i * args_for_me->my_grid->dimension + j;
					args_for_me->my_grid->element[pos] = \
						0.20*(args_for_me->temp->element[pos] + \
						   args_for_me->temp->element[(i - 1) * args_for_me->temp->dimension + j] +\
						   args_for_me->temp->element[(i + 1) * args_for_me->temp->dimension + j] +\
						   args_for_me->temp->element[i * args_for_me->temp->dimension + (j + 1)] +\
						   args_for_me->temp->element[i * args_for_me->temp->dimension + (j - 1)]);
					local_diff += fabs(args_for_me->my_grid->element[pos] - args_for_me->temp->element[pos]); 
				}
			}
		}
		/* Lock to write to shared difference */
		pthread_mutex_lock(&mutex);
		diff += local_diff;
		pthread_mutex_unlock(&mutex);
		
		 if((float)diff/((float)(args_for_me->my_grid->dimension*args_for_me->my_grid->dimension)) < (float)TOLERANCE) 
            		done = 1;

		/* Barrier sync and reset difference */
		barrier_sync(&barrier);		
	}
}

void * red_black(void *args){
	ARGS_FOR_THREAD *args_for_me = (ARGS_FOR_THREAD *)args;
	
	diff = 0;
	num_iter = 0;

	float local_diff;
	float temp;
	
	while(done != 1){
		local_diff = 0;
		for(int i = args_for_me->thread_id + 1; i < args_for_me->my_grid->dimension-1; i+=NUM_THREADS){
			if(is_red == 0){ /* Operate on only red points */
				if(args_for_me->thread_id % 2 == 0){ /* Alternates positions starting with Red or Black */
					for(int j = 1; j < args_for_me->my_grid->dimension-1; j+=2){
						/* Advance by 2 points to jump over the black point */
						int pos = i * args_for_me->my_grid->dimension + j;
						temp = args_for_me->my_grid->element[pos];
						args_for_me->my_grid->element[pos] = \
							0.20*(args_for_me->my_grid->element[pos] + \
							   args_for_me->my_grid->element[(i - 1) * args_for_me->my_grid->dimension + j] +\
							   args_for_me->my_grid->element[(i + 1) * args_for_me->my_grid->dimension + j] +\
							   args_for_me->my_grid->element[i * args_for_me->my_grid->dimension + (j + 1)] +\
							   args_for_me->my_grid->element[i * args_for_me->my_grid->dimension + (j - 1)]);
						local_diff += fabs(args_for_me->my_grid->element[i * args_for_me->my_grid->dimension + j] - temp); 
					}	
				}
				else{ /* Starts one position in to start on Red */
					for(int j = 2; j < args_for_me->my_grid->dimension-1; j+=2){
						int pos = i * args_for_me->my_grid->dimension + j;
						temp = args_for_me->my_grid->element[pos];
						args_for_me->my_grid->element[pos] = \
							0.20*(args_for_me->my_grid->element[pos] + \
							   args_for_me->my_grid->element[(i - 1) * args_for_me->my_grid->dimension + j] +\
							   args_for_me->my_grid->element[(i + 1) * args_for_me->my_grid->dimension + j] +\
							   args_for_me->my_grid->element[i * args_for_me->my_grid->dimension + (j + 1)] +\
							   args_for_me->my_grid->element[i * args_for_me->my_grid->dimension + (j - 1)]);
						local_diff += fabs(args_for_me->my_grid->element[i * args_for_me->my_grid->dimension + j] - temp); 
					}
				}
			}
			else{ /* Operate on only black points */
				if(args_for_me->thread_id % 2 == 0){ /* Starts one position in to start on black point */
					for(int j = 2; j < args_for_me->my_grid->dimension-1; j+=2){
						int pos = i * args_for_me->my_grid->dimension + j;
						temp = args_for_me->my_grid->element[pos];
						args_for_me->my_grid->element[pos] = \
							0.20*(args_for_me->my_grid->element[pos] + \
							   args_for_me->my_grid->element[(i - 1) * args_for_me->my_grid->dimension + j] +\
							   args_for_me->my_grid->element[(i + 1) * args_for_me->my_grid->dimension + j] +\
							   args_for_me->my_grid->element[i * args_for_me->my_grid->dimension + (j + 1)] +\
							   args_for_me->my_grid->element[i * args_for_me->my_grid->dimension + (j - 1)]);
						local_diff += fabs(args_for_me->my_grid->element[i * args_for_me->my_grid->dimension + j] - temp); 
					}	
				}
				else{
					for(int j = 1; j < args_for_me->my_grid->dimension; j+=2){
						int pos = i * args_for_me->my_grid->dimension + j;
						temp = args_for_me->my_grid->element[pos];
						args_for_me->my_grid->element[pos] = \
							0.20*(args_for_me->my_grid->element[pos] + \
							   args_for_me->my_grid->element[(i - 1) * args_for_me->my_grid->dimension + j] +\
							   args_for_me->my_grid->element[(i + 1) * args_for_me->my_grid->dimension + j] +\
							   args_for_me->my_grid->element[i * args_for_me->my_grid->dimension + (j + 1)] +\
							   args_for_me->my_grid->element[i * args_for_me->my_grid->dimension + (j - 1)]);
						local_diff += fabs(args_for_me->my_grid->element[i * args_for_me->my_grid->dimension + j] - temp); 
					}
				}
			}
			
		}
		/* Lock and store shared difference */
		pthread_mutex_lock(&mutex);
		diff += local_diff;
		pthread_mutex_unlock(&mutex);

		if((float)diff/((float)(args_for_me->my_grid->dimension*args_for_me->my_grid->dimension)) < (float)TOLERANCE) 
           		 done = 1;

		/* Barrier sync, reset difference and alternate red/black */
		barrier_sync(&barrier);
	}	
}

/* Edit this function to use the red-black method of solving the equation. The final result 
 *  * should be placed in the grid_3 data structure */
int 
compute_using_pthreads_red_black(GRID_STRUCT *grid_3)
{		
	pthread_t worker_thread[NUM_THREADS];
	ARGS_FOR_THREAD *args_for_thread;

	int i;

	for( i = 0; i < NUM_THREADS; i++){
		args_for_thread = (ARGS_FOR_THREAD *)malloc(sizeof(ARGS_FOR_THREAD));
		args_for_thread->thread_id = i;
		args_for_thread->my_grid = grid_3;

		if(pthread_create(&worker_thread[i], NULL, red_black, (void *) args_for_thread) != 0){
			printf("ERROR");
			exit(0);
		}

	}

	for(i = 0; i < NUM_THREADS; i++)
		pthread_join(worker_thread[i],NULL);

	return num_iter;		
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
	printf("Using the single threaded version to solve the grid. \n");
	int num_iter = compute_gold(grid_1);
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
		if(is_red == 1) //Change which set of points to operate on
			is_red = 0;
		else
			is_red = 1;
		num_iter++;
        	printf("Iteration %d. Diff: %f. \n", num_iter, diff);
		diff = 0; // Reset difference for new iteration
		pthread_cond_broadcast(&(barrier->condition)); 
	}
	else
	{
		while((pthread_cond_wait(&(barrier->condition), &(barrier->mutex))) != 0);
	}

	pthread_mutex_unlock(&(barrier->mutex));
}

