#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#ifndef NUMT
#define NUMT 2
#endif

// Increased trials to 500,000 for better probability resolution
#ifndef NUMTRIALS
#define NUMTRIALS 500000 
#endif

#define NUMTIMES 20

#define CSV

// Bend, Oregon Historical First Snow Data (Day of the Year)
// November 9th is Day 313
const float MEAN_SNOW_DAY = 313.f; 
const float STD_DEV       = 15.f;

// September 24th (Standard day of the year)
const float EARLIEST_HISTORICAL_DAY = 267.f; 

// May 23rd of the following spring ( 365 + 143 )
const float LATEST_HISTORICAL_DAY   = 508.f;

// Target: Thanksgiving Day (Day 331)
const int TARGET_DAY = 331;

float U1[NUMTRIALS];
float U2[NUMTRIALS];

// Function for getting a random number between two floats
float Ranf( float low, float high ) {
    float r = (float) rand();
    return low + r * ( high - low ) / (float) RAND_MAX;
}

// Call this if you want program to use a different random number sequence each time you run it
void TimeOfDaySeed() {
    struct tm y2k = { 0 };
    y2k.tm_hour = 0;   y2k.tm_min = 0; y2k.tm_sec = 0;
    y2k.tm_year = 100; y2k.tm_mon = 0; y2k.tm_mday = 1;

    time_t  timer;
    time( &timer );
    double seconds = difftime( timer, mktime(&y2k) );
    unsigned int seed = (unsigned int)( 1000.*seconds );
    srand( seed );
}

int main( int argc, char *argv[ ] ) {
    TimeOfDaySeed();

    omp_set_num_threads( NUMT );

    // Pre-fill uniform random arrays to avoid OpenMP thread-safety issues with rand()
    for( int n = 0; n < NUMTRIALS; n++ ) {
        U1[n] = Ranf( 0.f, 1.f );
        U2[n] = Ranf( 0.f, 1.f );
    }

    double maxPerformance = 0.;
    int numSuccesses;

    for( int times = 0; times < NUMTIMES; times++ ) {
        double time0 = omp_get_wtime();
        numSuccesses = 0;

        #pragma omp parallel for reduction(+:numSuccesses)
        for( int n = 0; n < NUMTRIALS; n++ ) {
            float u1 = U1[n];
            float u2 = U2[n];

            // Prevent log(0) in the equation
            if( u1 == 0.f ) u1 = 0.0001f;

            // Box-Muller transform: converts 2 uniform variables into 1 normal variable
            float z0 = sqrt( -2.0 * log( u1 ) ) * cos( 2.0 * M_PI * u2 );
            
            // Apply Bend's historical climate statistics
            float simulated_day = MEAN_SNOW_DAY + ( z0 * STD_DEV );
            
            // Skip this simulation if the simulated day is outside of historic limits
            if( simulated_day < EARLIEST_HISTORICAL_DAY ||simulated_day > LATEST_HISTORICAL_DAY ) {
                continue; // Skips to the next iteration without counting it as a success
            }

            // Did the first snow fall exactly on our target day?
            if( (int)simulated_day == TARGET_DAY ) {
                numSuccesses++;
            }
        }

        double time1 = omp_get_wtime();
        double megaTrialsPerSecond = (double)NUMTRIALS / ( time1 - time0 ) / 1000000.;
        if( megaTrialsPerSecond > maxPerformance )
            maxPerformance = megaTrialsPerSecond;
    }

    float probability = (float)numSuccesses / (float)NUMTRIALS;

#ifdef CSV
    fprintf(stderr, "%2d, %8d, %6.2lf, %6.2f\n", NUMT, NUMTRIALS, maxPerformance,100.*probability);
#else    
    fprintf(stderr, "%2d threads : %8d trials ;  megatrials/sec = %6.2lf ; probability = %6.2f%% ;\n", NUMT, NUMTRIALS, maxPerformance, 100.*probability);
#endif
    return 0;
}
