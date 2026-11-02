// Madison Neiswonger
//Project 5 - Fall 2026 

// Monte Carlo simulation of first snow probability (CUDA):

// system includes
#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <malloc.h>

// CUDA runtime
#include <cuda_runtime.h>

// setting the number of trials in the monte carlo simulation:
#ifndef NUMTRIALS
#define NUMTRIALS	500000 
#endif

#ifndef BLOCKSIZE
#define BLOCKSIZE	64     // number of threads per block
#endif

#define NUMBLOCKS	( NUMTRIALS / BLOCKSIZE )

#define CSV

// Device constants mapped from the original snow.cpp
__device__ const float MEAN_SNOW_DAY           = 313.f; 
__device__ const float STD_DEV                 = 15.f;
__device__ const float EARLIEST_HISTORICAL_DAY = 267.f; 
__device__ const float LATEST_HISTORICAL_DAY   = 508.f;
__device__ const int   TARGET_DAY              = 331;
__device__ const float CUDA_PI                 = 3.14159265358979323846f;

// Host arrays
float hU1[NUMTRIALS];
float hU2[NUMTRIALS];
int   hsuccesses[NUMTRIALS];

float
Ranf(float low, float high)
{
	float r = (float)rand();             // 0 - RAND_MAX
	float t = r / (float)RAND_MAX;       // 0. - 1.

	return   low + t * (high - low);
}

// call this if you want to force your program to use
// a different random number sequence every time you run it:
void
TimeOfDaySeed()
{
	struct tm y2k = { 0 };
	y2k.tm_hour = 0;   y2k.tm_min = 0; y2k.tm_sec = 0;
	y2k.tm_year = 100; y2k.tm_mon = 0; y2k.tm_mday = 1;

	time_t  timer;
	time(&timer);
	double seconds = difftime(timer, mktime(&y2k));
	unsigned int seed = (unsigned int)(1000. * seconds);    // milliseconds
	srand(seed);
}

void
CudaCheckError()
{
	cudaError_t e = cudaGetLastError();
	if (e != cudaSuccess)
	{
		fprintf(stderr, "Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));
	}
}

#define IN
#define OUT

__global__
void
MonteCarlo( IN float* dU1, IN float* dU2, OUT int* dsuccesses)
{
	// "gid", the global identifier, is essentially the for-loop index:
	unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

	dsuccesses[gid] = 0;

        // Load this threads u1 and u2
	float u1 = dU1[gid];
	float u2 = dU2[gid];

	// Prevent log(0) in the equation
	if( u1 == 0.f ) u1 = 0.0001f;

	// Box-Muller transform using CUDA device math functions
	float z0 = sqrtf( -2.0f * logf( u1 ) ) * cosf( 2.0f * CUDA_PI * u2 );
            
	// Apply Bend's historical climate statistics
	float simulated_day = MEAN_SNOW_DAY + ( z0 * STD_DEV );
	
	// Determine if this simulated snow day is at or before our target day taking the historical limits into account
        // dsuccesses[gid] is your output for this thread's simulated day.
        if ( simulated_day >= EARLIEST_HISTORICAL_DAY && simulated_day <= TARGET_DAY )
            dsuccesses[gid] = 1;
}

int
main(int argc, char* argv[])
{
	TimeOfDaySeed();               // seed the random number generator

	// fill the random-value arrays:
	for (int n = 0; n < NUMTRIALS; n++)
	{
		hU1[n] = Ranf(0.f, 1.f);
		hU2[n] = Ranf(0.f, 1.f);
	}

	// allocate device memory:
	float *dU1, *dU2;
	int   *dsuccesses;

	cudaMalloc((void**)(&dU1), NUMTRIALS * sizeof(float));
	cudaMalloc((void**)(&dU2), NUMTRIALS * sizeof(float));
	cudaMalloc((void**)(&dsuccesses), NUMTRIALS * sizeof(int));

	CudaCheckError();

	// copy host memory for U1 and U2 to the device:
	cudaMemcpy(dU1, hU1, NUMTRIALS * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dU2, hU2, NUMTRIALS * sizeof(float), cudaMemcpyHostToDevice);

	CudaCheckError();

	// setup the execution parameters:
	dim3 threads(BLOCKSIZE, 1, 1);
	dim3 grid(NUMBLOCKS, 1, 1);

	// create and start timer
	cudaDeviceSynchronize();

	// allocate CUDA events that we'll use for timing:
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	CudaCheckError();
	cudaEventCreate(&stop);
	CudaCheckError();

	// record the start event:
	cudaEventRecord(start, NULL);
	CudaCheckError();

	// execute the kernel:
	MonteCarlo<<< grid, threads >>>(dU1, dU2, dsuccesses);

	// record the stop event:
	cudaEventRecord(stop, NULL);

	// wait for the stop event to complete:
	cudaEventSynchronize(stop);

	float msecTotal = 0.0f;
	cudaEventElapsedTime(&msecTotal, start, stop);
	CudaCheckError();

	// copy result from the device to the host:
	cudaMemcpy(hsuccesses, dsuccesses, NUMTRIALS * sizeof(int), cudaMemcpyDeviceToHost);
	CudaCheckError();

	// compute the sum :
	int numSuccesses = 0;
	for (int i = 0; i < NUMTRIALS; i++)
	{
		numSuccesses += hsuccesses[i];
	}

	float probability = (float)numSuccesses / (float)NUMTRIALS;

	// compute and print the performance:
	double secondsTotal    = 0.001 * (double)msecTotal;
	double trialsPerSecond = (float)NUMTRIALS / secondsTotal;
	double megaTrialsPerSecond = trialsPerSecond / 1000000.;

#ifdef CSV
	fprintf(stderr, "%10d , %8d , %10.4lf , %6.2f\n",
		NUMTRIALS, BLOCKSIZE, megaTrialsPerSecond, 100.f * probability);
#else
	fprintf(stderr, "Number of Trials = %10d, Blocksize = %8d, MegaTrials/Second = %10.4lf, Probability = %6.2f%%\n",
		NUMTRIALS, BLOCKSIZE, megaTrialsPerSecond, 100. * probability);
#endif

	// clean up device memory:
	cudaFree(dU1);
	cudaFree(dU2);
	cudaFree(dsuccesses);

	CudaCheckError();

	return 0;
}