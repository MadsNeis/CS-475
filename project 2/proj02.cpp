// Madison Neiswonger
// CS 475 - Fall 2026
// Project 2


#include <stdio.h>
#include <stdlib.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

//Use this to disable barriers and compare to proper thread synchronization
//#define DISABLE_BARRIERS

omp_lock_t Lock;
volatile int NumInThreadTeam;
volatile int NumAtBarrier;
volatile int NumGone;
float Sqr(float x)
{
    return x * x;
}

void InitBarrier(int n)
{
    NumInThreadTeam = n; // number of threads you want to block at the barrier
    NumAtBarrier = 0;
    omp_init_lock(&Lock);
}
void WaitBarrier()
{
#ifndef DISABLE_BARRIERS
    omp_set_lock(&Lock);
    {
        NumAtBarrier++;
        if (NumAtBarrier == NumInThreadTeam) // release the waiting threads
        {
            NumGone = 0;
            NumAtBarrier = 0;
            // let all other threads return before this one unlocks:
            while (NumGone != NumInThreadTeam - 1)
                ;
            omp_unset_lock(&Lock);
            return;
        }
    }
    omp_unset_lock(&Lock);
    while (NumAtBarrier != 0)
        ;           // all threads wait here until the last one arrives â€¦
#pragma omp atomic // â€¦ and sets NumAtBarrier to 0
    NumGone++;
#endif
}

float Ranf(float low, float high)
{
    float r = (float)rand(); // 0 - RAND_MAX

    return (low + r * (high - low) / (float)RAND_MAX);
}


int	NowYear;		// 2024- 2030
int	NowMonth;	   // 0 - 11

float	NowPrecip;		  // inches of rain per month
float	NowTemp;		 // temperature this month
float	NowHeight;		// grain height in inches
float NowDisease;       // added variable : disease level in the deer herd, 0 = none and 1 = outbreak
int	NowNumDeer;		   // number of deer in the current population

const float GRAIN_GROWS_PER_MONTH =	       12.0;
const float ONE_DEER_EATS_PER_MONTH =		1.0;

const float AVG_PRECIP_PER_MONTH =		7.0;	 // average
const float AMP_PRECIP_PER_MONTH =		6.0;	// plus or minus
const float RANDOM_PRECIP =			2.0;	   // plus or minus noise

const float AVG_TEMP =				60.0;	  // average
const float AMP_TEMP =				20.0;	 // plus or minus
const float RANDOM_TEMP =			10.0;	// plus or minus noise

const float MIDTEMP =				40.0;
const float MIDPRECIP =				10.0;

// constants for added disease variable (insect borne disease in the deer, like EHD / bluetongue):
const float DISEASE_DEATH_RATE = 0.5;     // fraction of deer lost per month at full disease level (set to 0 to turn the disease off)
const float DISEASE_FROST_TEMP = 45.0;   // a monthly average below this means frosts, which kill the insects that spread it
const float DISEASE_CARRYOVER = 0.6;    // fraction of last month's disease level that carries over
const float DISEASE_CROWDING = 4.0;    // deer count at which contact between deer is "full"

void ComputeWeather()
{
    float ang = (30. * (float)NowMonth + 15.) * (M_PI / 180.);

    float temp = AVG_TEMP - AMP_TEMP * cos(ang);
    NowTemp = temp + Ranf(-RANDOM_TEMP, RANDOM_TEMP);

    float precip = AVG_PRECIP_PER_MONTH + AMP_PRECIP_PER_MONTH * sin(ang);
    NowPrecip = precip + Ranf(-RANDOM_PRECIP, RANDOM_PRECIP);
    if (NowPrecip < 0.)
    NowPrecip = 0.;
}

void Watcher()
{
    while (NowYear < 2030)
    {
        // do nothing
        WaitBarrier(); // 1.
        // do nothing
        WaitBarrier(); // 2.

        int monthNum = (NowYear - 2024) * 12 + NowMonth;
        float tempC = (5. / 9.) * (NowTemp - 32.);
        float precipCM = NowPrecip * 2.54;
        float heightCM = NowHeight * 2.54;
        printf("%d, %d, %d, %.2f, %.2f, %.2f, %d, %.3f\n",
               monthNum, NowYear, NowMonth + 1, tempC, precipCM, heightCM, NowNumDeer, NowDisease);

        // Advance time
        NowMonth++;
        if (NowMonth > 11)
        {
            NowYear++;
            NowMonth = 0;

        }
        
        // Re compute environment variables for new month
        ComputeWeather();
 
        WaitBarrier();
    }

}

void Deer()
{
    while (NowYear < 2030)
    {
        int nextNumDeer = NowNumDeer;
        int carryingCapacity = (int)( NowHeight );
        if( nextNumDeer < carryingCapacity )
                nextNumDeer++;
        else
        if( nextNumDeer > carryingCapacity )
                nextNumDeer--;

        // Disease kills deer (more deer in contact with each other means more spreading)
        float contact = (float)NowNumDeer / DISEASE_CROWDING;
        if (contact > 1.)
            contact = 1.;
        nextNumDeer -= (int)((float)NowNumDeer * DISEASE_DEATH_RATE * NowDisease * contact + 0.5);
        
        if (nextNumDeer < 0)
            nextNumDeer = 0;

        WaitBarrier(); // 1.

        // Copy the computed next state to the Now state
        NowNumDeer = nextNumDeer;

        WaitBarrier(); // 2.
        // // Do nothing
        WaitBarrier(); // 3.
    }
}

void Grain()
{
    while (NowYear < 2030)
    {
        // Compute next state
        float tempFactor = exp(-Sqr((NowTemp - MIDTEMP) / 10.));

        float precipFactor = exp(-Sqr((NowPrecip - MIDPRECIP) / 10.));

        float nextHeight = NowHeight;
        nextHeight += tempFactor * precipFactor * GRAIN_GROWS_PER_MONTH;
        nextHeight -= (float)NowNumDeer * ONE_DEER_EATS_PER_MONTH;
        if (nextHeight < 0.) {
            nextHeight = 0.;
        }
        WaitBarrier(); // 1.
        // Copy the computed next state to the Now state
        NowHeight = nextHeight;

        WaitBarrier(); // 2.
        // Do nothing
        WaitBarrier(); // 3.
    }
}

// Added Agent: disease (spread by bug / insect bites, like EHD / bluetongue.)
// it reads from simulation: temperature (warm months helps insects that spread it, 
// a frost kills them. Number of deer is read inside Deer() where deaths are applied
// effect on the simulation: sick deer die (see Deer()), which leaves more grain

void Disease()
{
    while (NowYear < 2030)
    {
        // a frost month wipes it out
        float nextDisease = 0.; 
        if (NowTemp >= DISEASE_FROST_TEMP)

            // warmer = faster spread (full speed at 80F)
            nextDisease = NowDisease * DISEASE_CARRYOVER + 0.5 * (NowTemp - DISEASE_FROST_TEMP) / 35.;

            if (nextDisease > 1.)
            nextDisease = 1.;
 
        WaitBarrier();
 
        NowDisease = nextDisease;
 
        WaitBarrier();
        // Do nothing
        WaitBarrier();
    }
}


int main(int argc, char *argv[])
{
    printf("MonthNum, Year, Month, TempC, PrecipCM, GrainHeightCM, NumDeer, Disease\n");

    // starting date and time:
    NowMonth =    0;
    NowYear  = 2024;

    // starting state (feel free to change this if you want):
    NowNumDeer = 2;
    NowHeight =  5.;
    NowDisease = 0.;

    ComputeWeather();

    InitBarrier(4);
    omp_set_num_threads(4); // same as # of sections
#pragma omp parallel sections
    {
#pragma omp section
        {
            Deer();
        }

#pragma omp section
        {
            Grain();
        }

#pragma omp section
        {
            Watcher();
        }

#pragma omp section
        {
            Disease(); // your own
        }
    } // implied barrier -- all functions must return in order
      // to allow any of them to get past here
    return 0;
}