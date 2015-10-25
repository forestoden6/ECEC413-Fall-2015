/* Pthread Lab: Histogram generation
 * Author: Forest Oden
 * Date modified: 10/24/2015
 *  *
 *   * compile as follows: 
 *    * gcc -o histogram histogram.c -std=c99 -lpthread -lm
 *     */
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <math.h>
#include <float.h>
#include <pthread.h>

void run_test(int);
void compute_gold(int *, int *, int, int);
void compute_using_pthreads(int *, int *, int, int);
void check_histogram(int *, int, int);
void* hist(void*);

#define HISTOGRAM_SIZE 500      /* Number of histrogram bins. */
#define NUM_THREADS  4          /* Number of threads. */

typedef struct args_for_thread_t{
	int threadID;
	int *histogram_data;
	int num_elements;
	int * input_data;
	int histogram_size;
} ARGS_FOR_THREAD;

pthread_mutex_t mutex;

int 
main( int argc, char** argv) 
{
	if(argc != 2){
		printf("Usage: histogram <num elements> \n");
		exit(0);	
	}
	int num_elements = atoi(argv[1]);
	run_test(num_elements);
	return 0;
}

void 
run_test(int num_elements) 
{
	float diff;
	int i; 

    /* Allocate memory for the histrogram structures. */
	int *reference_histogram = (int *)malloc(sizeof(int) * HISTOGRAM_SIZE);
	int *histogram_using_pthreads = (int *)malloc(sizeof(int) * HISTOGRAM_SIZE); 

	/* Generate input data---integer values between 0 and (HISTOGRAM_SIZE - 1). */
    int size = sizeof(int) * num_elements;
	int *input_data = (int *)malloc(size);

	for(i = 0; i < num_elements; i++)
		input_data[i] = floorf((HISTOGRAM_SIZE - 1) * (rand()/(float)RAND_MAX));

    /* Compute the reference solution on the CPU. */
	printf("Creating the reference histogram. \n"); 
	struct timeval start, stop;	
	gettimeofday(&start, NULL);

	compute_gold(input_data, reference_histogram, num_elements, HISTOGRAM_SIZE);

	gettimeofday(&stop, NULL);
	printf("Singlethreaded CPU run time = %0.2f s. \n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));
	check_histogram(reference_histogram, num_elements, HISTOGRAM_SIZE); 
	
	/* Compute the histogram using pthreads. The result histogram should be stored in the 
 *      * histogram_using_pthreads array. */
	printf("Creating histogram using pthreads. \n");

	gettimeofday(&start, NULL);
	compute_using_pthreads(input_data, histogram_using_pthreads, num_elements, HISTOGRAM_SIZE);
	gettimeofday(&stop, NULL);
	printf("Multithreaded CPU run time = %0.2f s. \n", (float) (stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));

	check_histogram(histogram_using_pthreads, num_elements, HISTOGRAM_SIZE); 
	/* Compute the differences between the reference and pthread results. */
	diff = 0.0;
    for(i = 0; i < HISTOGRAM_SIZE; i++)
		diff = diff + abs(reference_histogram[i] - histogram_using_pthreads[i]);

	printf("Difference between the reference and pthread results: %f. \n", diff);
   
	/* cleanup memory. */
	free(input_data);
	free(reference_histogram);
	free(histogram_using_pthreads);

	pthread_exit(NULL);
}

/* This function computes the reference solution. */
void 
compute_gold(int *input_data, int *histogram, int num_elements, int histogram_size)
{
  int i;
  
   for(i = 0; i < histogram_size; i++)   /* Initialize histogram. */
       histogram[i] = 0; 

   for(i = 0; i < num_elements; i++)     /* Bin the elements. */
			 histogram[input_data[i]]++;
}


/* Write the function to compute the histogram using pthreads. */
void 
compute_using_pthreads(int *input_data, int *histogram, int num_elements, int histogram_size)
{
	pthread_t threads[NUM_THREADS];
	ARGS_FOR_THREAD *args_for_thread;
	
	int i;
	
	/* Give each thread its private variables and create thread  */
	for(i = 0; i < NUM_THREADS; i++){
		args_for_thread = (ARGS_FOR_THREAD *)malloc(sizeof(ARGS_FOR_THREAD));
		args_for_thread->threadID = i;
		args_for_thread->histogram_data = histogram;
		args_for_thread->num_elements = num_elements;
		args_for_thread->input_data = input_data;
		args_for_thread->histogram_size = histogram_size;

		if(pthread_create(&threads[i], NULL, hist, (void *) args_for_thread)!= 0) {
			printf("Cannot create thread\n");
			exit(0);
			}
	}

	/* Wait for threads to finish */
	for(int i = 0; i < NUM_THREADS; i++) {
		pthread_join(threads[i],NULL);
	}		
}

/*Histogram generation logic for each thread. */
void*
hist(void *this_arg) {

	ARGS_FOR_THREAD *args_for_me = (ARGS_FOR_THREAD *)this_arg;
	
	int temp_hist[args_for_me->histogram_size];
	int i;

	/* Initialize temporary histogram per thread */
	for(i = args_for_me->threadID; i < args_for_me->histogram_size; i++)
		temp_hist[i]=0;
	/* Start at thread ID and go through input data by jumping NUM_THREADS to get part of the full data  */
	for(i = args_for_me->threadID; i < args_for_me->num_elements; i += NUM_THREADS)
		temp_hist[args_for_me->input_data[i]] += 1;
		
	/* For the entire histogram, lock and add the temp histogram to the shared histogram  */
	for(i = 0; i < args_for_me->histogram_size; i++)
	{	if(temp_hist[i] > 0){
			pthread_mutex_lock(&mutex);
			args_for_me->histogram_data[i] += temp_hist[i];
			pthread_mutex_unlock(&mutex);
		}
	}
}
/* Helper function to check for correctness of the resulting histogram. */
void 
check_histogram(int *histogram, int num_elements, int histogram_size)
{
	int sum = 0;
	for(int i = 0; i < histogram_size; i++)
		sum += histogram[i];

	printf("Number of histogram entries = %d. \n", sum);
	if(sum == num_elements)
		printf("Histogram generated successfully. \n");
	else
		printf("Error generating histogram. \n");
}

