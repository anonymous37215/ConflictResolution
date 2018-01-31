#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <math.h>
#include <curand_kernel.h>
#include <time.h>
#include <unistd.h>
#include <thrust/scan.h>
#include <thrust/remove.h>
#include <thrust/execution_policy.h>
#include <iostream>
#include "custom_temporary_allocation.cuh"
#include "parameter.cuh"
using namespace std;

typedef curandStatePhilox4_32_10_t myCurandState_t;

//#define DEBUG

#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   exit(0); \
 }                                                                 \
}
 
#define FACTOR 5
#define ITERATIONS 100
#define TOTAL (SIZE * SIZE)

#define GRIDSIZE (SIZE+2)
#define GRIDTOTAL (SIZE+2)*(SIZE+2)
#define SRAND_VALUE 200
#define PENUMBER (TOTAL/PESIZE)

#define SAM_NUM_VALUES ((SIZE+2)*(SIZE+2))
#define SAM_PENUMBER (SAM_NUM_VALUES / SAM_PESIZE)

const int agentTypeOneNumber = agentNumber / 2;
const int agentTypeTwoNumber = agentNumber - agentTypeOneNumber;
const int happinessThreshold = 5;

void printOutput(int [SIZE+2][SIZE+2]);
void initPos(int grid [SIZE+2][SIZE+2]);
int random_location();

__device__ static const int FAK_LEN = 1024;       // length of factorial table
__device__ int  hyp_n_last[SAM_PENUMBER], hyp_m_last[SAM_PENUMBER], hyp_N_last[SAM_PENUMBER];            // Last values of parameters
__device__ int  hyp_mode[SAM_PENUMBER], hyp_mp[SAM_PENUMBER];                              // Mode, mode+1
__device__ int  hyp_bound[SAM_PENUMBER];                                     // Safety upper bound
__device__ double hyp_a[SAM_PENUMBER];                                           // hat center
__device__ double hyp_h[SAM_PENUMBER];                                           // hat width
__device__ double hyp_fm[SAM_PENUMBER];                                          // Value at mode


__device__ int device_pe_inuse;
__device__ int device_num_inuse;


__device__ int device_removed_move_list_end;
__device__ int device_removed_space_list_end;

__device__ int device_penumber_inuse;
__device__ int device_reduced_pe_position;

__device__ float getnextrand(myCurandState_t *state){

	
	return (curand_uniform(state));
}


__global__ void initSamCurand(myCurandState_t state[SAM_PENUMBER]){


	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if(idx < SAM_PENUMBER){
		curand_init(idx, 0 , 0, &state[idx]);
	}
}

__device__ const double     
	   C0 =  0.918938533204672722,      // ln(sqrt(2*pi))
	   C1 =  1./12.,
	   C3 = -1./360.;

__device__ double fac_table[FAK_LEN];
__device__ int initialized = 0; 

__device__ double LnFac(int n) {

   if (n < FAK_LEN) {
      if (n <= 1) {
         if (n < 0) printf("Parameter negative in LnFac function\n");  
         return 0;
      }
      if (!initialized) {              // first time. Must initialize table
         // make table of ln(n!)
         double sum = fac_table[0] = 0.;
         for (int i=1; i<FAK_LEN; i++) {
            sum += log(double(i));
            fac_table[i] = sum;
         }
         initialized = 1;
      }
      return fac_table[n];
   }
   // not found in table. use Stirling approximation
   double  n1, r;
   n1 = n;  r  = 1. / n1;
   return (n1 + 0.5)*log(n1) - n1 + C0 + r*(C1 + r*r*C3);

	//return logf(n);
}




__device__  double fc_lnpk(int k, int L, int m, int n) {
   // subfunction used by hypergeometric and Fisher's noncentral hypergeometric distribution
   return(LnFac(k) + LnFac(m - k) + LnFac(n - k) + LnFac(L + k));
}




__device__ int HypInversionMod (myCurandState_t stateHyper[SAM_PENUMBER],int n, int m, int N, int idx) {
   /* 
   Subfunction for Hypergeometric distribution. Assumes 0 <= n <= m <= N/2.
   Overflow protection is needed when N > 680 or n > 75.

   Hypergeometric distribution by inversion method, using down-up 
   search starting at the mode using the chop-down technique.

   This method is faster than the rejection method when the variance is low.
   */
   //int idx = threadIdx.x + blockIdx.x * blockDim.x;
   // Sampling 
   int       I;                    // Loop counter
   int       L = N - m - n;        // Parameter
   double        modef;                // mode, float
   double        Mp, np;               // m + 1, n + 1
   double        p;                    // temporary
   double        U;                    // uniform random
   double        c, d;                 // factors in iteration
   double        divisor;              // divisor, eliminated by scaling
   double        k1, k2;               // float version of loop counter
   double        L1 = L;               // float version of L

   Mp = (double)(m + 1);
   np = (double)(n + 1);

   if (N != hyp_N_last[idx] || m != hyp_m_last[idx] || n != hyp_n_last[idx]) {
      // set-up when parameters have changed
      hyp_N_last[idx] = N;  hyp_m_last[idx] = m;  hyp_n_last[idx] = n;

      p  = Mp / (N + 2.);
      modef = np * p;                       // mode, real
      hyp_mode[idx] = (int)modef;            // mode, integer
      if (hyp_mode[idx] == modef && p == 0.5) {   
         hyp_mp[idx] = hyp_mode[idx]--;
      }
      else {
         hyp_mp[idx] = hyp_mode[idx] + 1;
      }
      // mode probability, using log factorial function
      // (may read directly from fac_table if N < FAK_LEN)
      hyp_fm[idx] = exp(LnFac(N-m) - LnFac(L+hyp_mode[idx]) - LnFac(n-hyp_mode[idx])
         + LnFac(m)   - LnFac(m-hyp_mode[idx]) - LnFac(hyp_mode[idx])
         - LnFac(N)   + LnFac(N-n)      + LnFac(n)        );

      // safety bound - guarantees at least 17 significant decimal digits
      // bound = min(n, (int)(modef + k*c'))
      hyp_bound[idx] = (int)(modef + 11. * sqrt(modef * (1.-p) * (1.-n/(double)N)+1.));
      if (hyp_bound[idx] > n) hyp_bound[idx] = n;
   }

   // loop until accepted


   //int max_iterations = 1000;
   while(1) {
     // if(!(max_iterations--))
      //    break;

      U = getnextrand(&stateHyper[idx]);                     // uniform random number to be converted
	//printf(" U is %lf\n",U);	
      // start chop-down search at mode
      if ((U -= hyp_fm[idx]) <= 0.) return(hyp_mode[idx]);
      c = d = hyp_fm[idx];

      // alternating down- and upward search from the mode
      k1 = hyp_mp[idx] - 1;  k2 = hyp_mode[idx] + 1;
      for (I = 1; I <= hyp_mode[idx]; I++, k1--, k2++) {
       //  if(!(max_iterations--))
       //      break;

         // Downward search from k1 = hyp_mp - 1
         divisor = (np - k1)*(Mp - k1);
         // Instead of dividing c with divisor, we multiply U and d because 
         // multiplication is faster. This will give overflow if N > 800
         U *= divisor;  d *= divisor;
         c *= k1 * (L1 + k1);
         if ((U -= c) <= 0.)  return(hyp_mp[idx] - I - 1); // = k1 - 1
	 //printf("Line 228 I %d \n",I);
         // Upward search from k2 = hyp_mode + 1
         divisor = k2 * (L1 + k2);
         // re-scale parameters to avoid time-consuming division
         U *= divisor;  c *= divisor; 
         d *= (np - k2) * (Mp - k2);
         if ((U -= d) <= 0.)  return(hyp_mode[idx] + I);  // = k2
         // Values of n > 75 or N > 680 may give overflow if you leave out this..
         // overflow protection
         // if (U > 1.E100) {U *= 1.E-100; c *= 1.E-100; d *= 1.E-100;}
      }

      // Upward search from k2 = 2*mode + 1 to bound
      for (k2 = I = hyp_mp[idx] + hyp_mode[idx]; I <= hyp_bound[idx]; I++, k2++) {
         //if(!(max_iterations--))
         //   break;

         divisor = k2 * (L1 + k2);
         U *= divisor;
         d *= (np - k2) * (Mp - k2);
         if ((U -= d) <= 0.)  return(I);
         // more overflow protection
         // if (U > 1.E100) {U *= 1.E-100; d *= 1.E-100;}
      }
   }
}



__device__ int HypRatioOfUnifoms (myCurandState_t stateHyper[SAM_PENUMBER], int n, int m, int N, int idx) {
   /*
   Subfunction for Hypergeometric distribution using the ratio-of-uniforms
   rejection method.

   This code is valid for 0 < n <= m <= N/2.

   The computation time hardly depends on the parameters, except that it matters
   a lot whether parameters are within the range where the LnFac function is
   tabulated.

   Reference: E. Stadlober: "The ratio of uniforms approach for generating
   discrete random variates". Journal of Computational and Applied Mathematics,
   vol. 31, no. 1, 1990, pp. 181-189.
   */
   //int idx = threadIdx.x + blockIdx.x * blockDim.x;
   const double SHAT1 = 2.943035529371538573;    // 8/e
   const double SHAT2 = 0.8989161620588987408;   // 3-sqrt(12/e)

   int L;                          // N-m-n
   int mode;                       // mode
   int k;                          // integer sample
   double x;                           // real sample
   double rNN;                         // 1/(N*(N+2))
   double my;                          // mean
   double var;                         // variance
   double u;                           // uniform random
   double lf;                          // ln(f(x))

   L = N - m - n;
   if (hyp_N_last[idx] != N || hyp_m_last[idx] != m || hyp_n_last[idx] != n) {
      hyp_N_last[idx] = N;  hyp_m_last[idx] = m;  hyp_n_last[idx] = n;         // Set-up
      rNN = 1. / ((double)N*(N+2));                             // make two divisions in one
      my = (double)n * m * rNN * (N+2);                         // mean = n*m/N
      mode = (int)(double(n+1) * double(m+1) * rNN * N);    // mode = floor((n+1)*(m+1)/(N+2))
      var = (double)n * m * (N-m) * (N-n) / ((double)N*N*(N-1));// variance
      hyp_h[idx] = sqrt(SHAT1 * (var+0.5)) + SHAT2;                  // hat width
      hyp_a[idx] = my + 0.5;                                         // hat center
      hyp_fm[idx] = fc_lnpk(mode, L, m, n);                          // maximum
      hyp_bound[idx] = (int)(hyp_a[idx] + 4.0 * hyp_h[idx]);               // safety-bound
      if (hyp_bound[idx] > n) hyp_bound[idx] = n;
   }    
   while(1) {

      u = getnextrand(&stateHyper[idx]);                            // uniform random number

      if (u == 0) continue;                      // avoid division by 0
      x = hyp_a[idx] + hyp_h[idx] * (getnextrand(&stateHyper[idx])-0.5) / u;    // generate hat distribution
      if (x < 0. || x > 2E9) continue;           // reject, avoid overflow
      k = (int)x;
      if (k > hyp_bound[idx]) continue;               // reject if outside range
      lf = hyp_fm[idx] - fc_lnpk(k,L,m,n);            // ln(f(k))
      if (u * (4.0 - u) - 3.0 <= lf) break;      // lower squeeze accept
      if (u * (u-lf) > 1.0) continue;            // upper squeeze reject
      if (2.0 * log(u) <= lf) break;             // final acceptance
   }
   return k;
}


__device__  int Hypergeometric (myCurandState_t stateHyper[SAM_PENUMBER], int n, int m, int N, int idx) {
   /*
   This function generates a random variate with the hypergeometric
   distribution. This is the distribution you get when drawing balls without 
   replacement from an urn with two colors. n is the number of balls you take,
   m is the number of red balls in the urn, N is the total number of balls in 
   the urn, and the return value is the number of red balls you get.

   This function uses inversion by chop-down search from the mode when
   parameters are small, and the ratio-of-uniforms method when the former
   method would be too slow or would give overflow.
   */   

   int fak, addd;                    // used for undoing transformations
   int x;                            // result



   hyp_n_last[idx] = hyp_m_last[idx] = hyp_N_last[idx] = -1; // Last values of hypergeometric parameters	


   // check if parameters are valid
   if (n > N || m > N || n < 0 || m < 0) {
      printf("Parameter out of range in hypergeometric function  n %ld m %ld N %ld idx %d\n",n,m,N,idx);
      printf("Parameter out of range in hypergeometric function  %d,%d,%d,%d\n", n > N, m > N, n < 0, m < 0);
      return 0;
   }
   // symmetry transformations
   fak = 1;  addd = 0;
   if (m > N/2) {
      // invert m
      m = N - m;
      fak = -1;  addd = n;
   }    
   if (n > N/2) {
      // invert n
      n = N - n;
      addd += fak * m;  fak = - fak;
   }    
   if (n > m) {
      // swap n and m
      x = n;  n = m;  m = x;
   }    
   // cases with only one possible result end here
   if (n == 0)  return addd;

   //------------------------------------------------------------------
   //                 choose method
   //------------------------------------------------------------------
   if (N > 680 || n > 70) {
      // use ratio-of-uniforms method
      x = HypRatioOfUnifoms (stateHyper, n, m, N,idx);
   }
   else {
      // inversion method, using chop-down search from mode
   x = HypInversionMod (stateHyper, n, m, N,idx);
   }
   // undo symmetry transformations  
   return x * fak + addd;
}


__global__ void clearSamples(int samples[SAM_NUM_VALUES]){

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if(idx  < (SAM_NUM_VALUES)){
		samples[idx] = 0;
	}

}
 
        
__device__ void methodA(myCurandState_t state[SAM_PENUMBER],int N, int n, int num_sample, int initialTocurrent,int device_list[SAM_NUM_VALUES],int samples[SAM_NUM_VALUES]) {
            //ASSERT_LEQ(n, N);
	    int idx = threadIdx.x + blockIdx.x * blockDim.x;
            // Initialization
            int sample = 0;
            double Nreal = (double) N;
            double top = Nreal - n;

            // Main loop
            while (n >= 2) {
                int S = 0;
                double V = getnextrand(&state[idx]);
                double quot = top / Nreal;
                while (quot > V) {
                    S++; 
                    top -= 1.0;
                    Nreal -= 1.0;
                    quot = (quot * top) / Nreal;
                }
                // Skip over next S records and select the following one
                sample += S + 1;
                //samples[idx][num_sample++] = sample + initialTocurrent;
		samples[idx*SAM_PESIZE + num_sample++] = device_list[idx*SAM_PESIZE + sample + initialTocurrent-1];
                //callback(sample);
                Nreal -= 1.0; 
                n--;
            }
            if (n == 1) {
                int S = round(Nreal) * getnextrand(&state[idx]);
                sample += S + 1;
                //samples[idx][num_sample++] = sample + initialTocurrent;
		samples[idx*SAM_PESIZE + num_sample++] = device_list[idx*SAM_PESIZE + sample + initialTocurrent-1];
                //callback(sample);
            }
        }

        // Sampling method D from Vitter et al.
        //
        // \param N Size of population.
        // \param n Number of samples.
        // \param gen Uniform random variate generator.
        // \param samples Function to process sample.
        // 


__device__ void sample(myCurandState_t state[SAM_PENUMBER], int N, int n, int device_list[SAM_NUM_VALUES], int samples[SAM_NUM_VALUES]) {
            //ASSERT_LEQ(n, N);
	    int idx = threadIdx.x + blockIdx.x * blockDim.x;
            int initialN = N;
            // Initialization
            int sample = 0;
             int num_sample = 0;
            double nreal = (double) n; 
            double ninv = 1.0 / nreal; 
            double Nreal = (double) N;
            double Vprime = exp(log(getnextrand(&state[idx])) * ninv);
            int qu1 = N + 1 - n; 
            double qu1real = Nreal + 1.0 - nreal;
            int negalphainv = -13; 
            int threshold = n * (-negalphainv);
            int S = 0;

            // Main loop
            while (n > 1 && threshold < N) {
                double nmin1inv = 1.0 / (nreal - 1.0);
                double negSreal = 0.0;

                while (true) {
                    // Step D2: Generate U and X
                    double X;
                    while (true) {
                        X = Nreal * (1.0 - Vprime); 
                        S = X;
                        if (S < qu1) break;
                        Vprime = exp(log(getnextrand(&state[idx])) * ninv);
                    }

                    double U = getnextrand(&state[idx]); 
                    negSreal = -(double)S;

                    // Step D3: Accept?
                    double y1 = exp(log(U * Nreal / qu1real) * nmin1inv);
                    Vprime = y1 * (-X / Nreal + 1.0) * (qu1real / (negSreal + qu1real));
                    if (Vprime <= 1.0) break; // Accept!

                    // Step D4: Accept?
                    double y2 = 1.0; double top = Nreal - 1.0;
                    double bottom;
                    double limit;
                    if (n - 1 > S) {
                        bottom = Nreal - nreal; 
                        limit = N - S;
                    } else {
                        bottom = negSreal + Nreal - 1.0;
                        limit = qu1;
                    }

                    for (int t = N; t > limit; t--) {
                        y2 = (y2 * top) / bottom;
                        top -= 1.0;
                        bottom -= 1.0;
                    }

                    if (Nreal / (Nreal - X) >= y1 * exp(log(y2) * nmin1inv)) {
                        // Accept!
                        Vprime = exp(log(getnextrand(&state[idx])) * nmin1inv);
                        break;
                    }
                    Vprime = exp(log(getnextrand(&state[idx])) * ninv);
                }
                // Skip over next S records and select the following one
                sample += S + 1;

                //samples[idx][num_sample++] = sample;
                samples[idx*SAM_PESIZE + num_sample++] = device_list[idx*SAM_PESIZE +sample-1];
                //callback(sample);
                N = (N - 1) - S;
                Nreal = (Nreal - 1.0) + negSreal;
                n--;
                nreal -= 1.0;
                ninv = nmin1inv;
                qu1 -= S;
                qu1real += negSreal;
                threshold += negalphainv;
            }

            if (n > 1) {
                int currentN = N;
		

		methodA(state, N, n, num_sample, initialN - currentN, device_list,samples);

		//samples[num_sample++] =  sample + initialN - currentN;


		
                //methodA(N, n, [&](int sample) {
                //        callback(sample + initialN - currentN);


                    //});
            } else if (n == 1) {
                S = N * Vprime;
                // Skip over next S records and select the following one
                sample += S + 1;

                //samples[idx][num_sample++] = sample;
                samples[idx*SAM_PESIZE + num_sample++] = device_list[idx*SAM_PESIZE +sample-1];
                //callback(sample);
            }
        }


__global__ void sampleP(myCurandState_t state[SAM_PENUMBER], myCurandState_t stateHyper[SAM_PENUMBER], int device_list[SAM_NUM_VALUES],int samples[SAM_NUM_VALUES], int n, int j, int k) {

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	//idx += 1;
	if(idx < device_pe_inuse){
		int seed = 1;
		//int counter = 0;
		int m,x;
		while(j - k != 0) {
			curand_init(seed, 0 , 0, &stateHyper[idx]);
			m = floor( (j+k)/2.0 );
			//printf("sampleP1 n %d idx %d m %d\n",n,idx,m);

	//__device__  int Hypergeometric (curandState stateHyper[PENUMBER], 
	//int n, int m, int N, int idx) {
	   /*
	   This function generates a random variate with the hypergeometric
	   distribution. This is the distribution you get when drawing balls without 
	   replacement from an urn with two colors. n is the number of balls you take,
	   m is the number of red balls in the urn, N is the total number of balls in 
	   the urn, and the return value is the number of red balls you get. */


			//printf("would call Hypergeometric(stateHyper, %d, %d, %d, %d)\n",  n, (m-j)*PESIZE + 1, (k-j)*PESIZE + 1, idx);
			//printf("j is now %d, k is %d, m is %d, sums are %d and %d\n", j, k, m, k - (j - 1), m - (j - 1));

			if(k != device_pe_inuse - 1){
				x  = Hypergeometric(stateHyper, n, (m-(j-1))*SAM_PESIZE, (k-(j-1))*SAM_PESIZE, idx);
			}

			else{
				
				x  = Hypergeometric(stateHyper, n, (m-(j-1))*SAM_PESIZE, ((k-1)-(j-1))*SAM_PESIZE + device_num_inuse % SAM_PESIZE, idx);

			}
			//printf("sampleP2 n %d idx %d  x %d\n",n,idx,x);

			//int x = m;
			if(idx <= m) {
		      		n = x;
		       		k = m;
		        
				seed = seed * 2;

		    	} else {
				n = n-x;
				j = m + 1;
				seed = seed * 2 + 1;
		    	}
		
		}
		//printf("sample n %d \n",n);



		if(idx != device_pe_inuse - 1 ) {
			//printf("idx %d sampling %d values\n", idx, n);
			sample(state, SAM_PESIZE, n, device_list, samples);
	 	}
		else {
			//printf("n > PESIZE  %d \n",n);
			sample(state, device_num_inuse % SAM_PESIZE, n, device_list, samples);
		}

		/*if(n <= PESIZE ) {
			//printf("idx %d sampling %d values\n", idx, n);
			sample(state, PESIZE, n, device_list, samples);
	 	}
		else {
			printf("n > PESIZE  %d \n",n);
		}*/

	}
}

//__global__ void print_device_reduced_pe_position(){


	//printf("reduced_pe_position %d \n",( int( 0.5 + ceil((float)device_reduced_pe_position / (PESIZE) )) ) );

	//printf("device_reduced_pe_position %d \n",(device_reduced_pe_position ) );

//}


__global__ void initCurand(myCurandState_t state[][PESIZE]){


	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int idy=blockIdx.y*blockDim.y+threadIdx.y;
	if(idx < PENUMBER && idy<PESIZE){
		curand_init(idx*(PESIZE)+idy,0 , 0, &state[idx][idy]);
	}
}


__global__ void compute(int grid[][SIZE+2], int new_grid[][SIZE+2], int * move_list, int * space_list, int iteration){
	
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	int idy=blockIdx.y*blockDim.y+threadIdx.y;
	int sameTypeCount=0;

	int current_id = idx*(SIZE+2)+idy;

	if(grid[idx][idy] != 0){

		int currentType = grid[idx][idy];

		if(grid[idx-1][idy-1] == currentType){
			sameTypeCount += 1;
		}

		if(grid[idx-1][idy] == currentType){
			sameTypeCount += 1;
		}

		if(grid[idx-1][idy+1] == currentType){
			sameTypeCount += 1;
		}

		if(grid[idx][idy-1] == currentType){
			sameTypeCount += 1;
		}

		if(grid[idx][idy+1] == currentType){
			sameTypeCount += 1;
		}

		if(grid[idx+1][idy-1] == currentType){
			sameTypeCount += 1;
		}

		if(grid[idx+1][idy] == currentType){
			sameTypeCount += 1;
		}

		if(grid[idx+1][idy+1] == currentType){
			sameTypeCount += 1;
		}

			
		if(sameTypeCount < happinessThreshold){
			
			move_list[current_id] = current_id;
			space_list[current_id] = current_id;


		}



	}
	else if(idx != 0 && idy !=0 && idx != (SIZE+1) && idy != (SIZE+1)  ){
		space_list[current_id] = current_id;

	}
	
}


__global__ void update (int grid[][SIZE+2], int new_grid[][SIZE+2],  int * move_list, int * space_list){

	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	int idy=blockIdx.y*blockDim.y+threadIdx.y;

	grid[idy][idx] = new_grid[idy][idx];

	move_list[idx*(SIZE+2)+idy] = 0;
	space_list[idx*(SIZE+2)+idy] = 0;
	

}




__global__ void sendToRandomPerpe(myCurandState_t state[][PESIZE],int device_list[SAM_NUM_VALUES], int temp_device_list[][PESIZE*FACTOR],int random_list_counter[PENUMBER]){

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if(idx < device_penumber_inuse -1 ){

		for(int i=0; i < PESIZE; i++ ){		

			float r = getnextrand(&state[idx][0]);
		
			int random_position =  r * (device_penumber_inuse-1);
		
			int acquired_position = atomicAdd(&random_list_counter[random_position],1);
	
			temp_device_list[random_position][acquired_position] = device_list[idx*PESIZE+i];
		}
	}


	else if(idx ==  device_penumber_inuse - 1 ){

		for(int i=0; i < device_removed_move_list_end % PESIZE; i++ ){		

			float r = getnextrand(&state[idx][0]);
		
			int random_position =  r * (device_penumber_inuse-1);
		
			int acquired_position = atomicAdd(&random_list_counter[random_position],1);

	
			temp_device_list[random_position][acquired_position] = device_list[idx*PESIZE+i];
		}


	}


}



__global__ void sendToRandom(myCurandState_t state[][PESIZE],int device_list[SAM_NUM_VALUES], int temp_device_list[][PESIZE*FACTOR],int random_list_counter[PENUMBER]){

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	if(idx*PESIZE +idy < device_removed_move_list_end ){
		
		float r = getnextrand(&state[idx][idy]);
		
		int random_position =  r * (device_penumber_inuse-1);
		
		int acquired_position = atomicAdd(&random_list_counter[random_position],1);
	
		temp_device_list[random_position][acquired_position] = device_list[idx*PESIZE+idy];
	}


}


__global__ void clearCounter(int random_list_counter[PENUMBER]){

	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if(idx < device_penumber_inuse){
		random_list_counter[idx] = 0;

	}
}


__global__ void generateList(int device_list[][PESIZE]){

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;

	if(idx*PESIZE +idy < device_removed_space_list_end ){	
		device_list[idx][idy] = idx*PESIZE +idy;
	}

}


static __device__ void swap(int *data, int x, int y)
{
  	int temp = data[x];
  	data[x] = data[y];
  	data[y] = temp;
}


static __device__ int partition(int *data, int left, int right)
{
	const int mid = left + (right - left) / 2;

 	const int pivot = data[(mid)];

  	swap(data, (mid), (left));

  	int i = left + 1;
  	int j = right;

  	while (i <= j) {
    		while (i <= j && data[(i)] <= pivot) {
      			i++;
    		}

    		while (i <= j && data[(j)] > pivot) {
      			j--;
    		}

    		if (i < j) {
      			swap(data, (i), (j));
    		}
  	}

  	swap(data, (i - 1), (left));
  	return i - 1;
}

typedef struct sort_data {
  	int left;
  	int right;
} sort_data;

__device__ void quicksort_seq(int *data, int right)
{
	int left = 0;

  	if(left == right)
    		return;

  	if (left > right) {
    		right = 1 + right;
  	}

 	int stack_size = 0;
  	sort_data stack[PESIZE*FACTOR];

  	stack[stack_size++] = { left, right }; 

	
  	while (stack_size > 0) {

    		int curr_left = stack[stack_size - 1].left;
    		int curr_right = stack[stack_size - 1].right;
    		stack_size--;

   	 	if (curr_left < curr_right) {
      			int part = partition(data, curr_left, curr_right);
      			stack[stack_size++] = {curr_left, part - 1};
      			stack[stack_size++] = {part + 1, curr_right};
    		}
  	}
}



__global__ void sortList(int temp_device_list[][PESIZE*FACTOR], int random_list_counter[PENUMBER]){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if(idx < device_penumber_inuse){

		int number = random_list_counter[idx];
		if(number != 0){

			quicksort_seq(temp_device_list[idx], number - 1 );
		}

	}

}

__global__ void randomPermute(myCurandState_t state[][PESIZE], int temp_device_list[][PESIZE*FACTOR], int random_list_counter[PENUMBER]){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	
	int reduced_pe = device_penumber_inuse;  
	if(idx < reduced_pe){
		for (int i = 0; i < random_list_counter[idx]; i++){
	    		

			float r = getnextrand(&state[idx][0]);		

	    		int j = r * (random_list_counter[idx]-1);
	    		
	   		int temp = temp_device_list[idx][i] ;
	   		temp_device_list[idx][i]  = temp_device_list[idx][j] ;
	    		temp_device_list[idx][j]  = temp;
		}	
	}

}



__global__ void recoverSize(int device_list[][PESIZE], int temp_device_list[][PESIZE*FACTOR],int random_list_counter[PENUMBER], int scanned_random_list_counter[PENUMBER]){

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int reduced_pe = device_penumber_inuse; 
	if(idx < reduced_pe){

		int delta = scanned_random_list_counter[idx];

		for(int i=0; i<random_list_counter[idx]; i++){
			int addValue = delta + i;
			int interResult = device_penumber_inuse*addValue/(PESIZE*device_penumber_inuse);

			device_list[interResult][(delta- (PESIZE*device_penumber_inuse/device_penumber_inuse)*interResult + i)]  = temp_device_list[idx][i];

		}
	}
	
}

struct smaller_than
{
  __device__
  bool operator()(const int x)
  {
    return (x < device_removed_space_list_end) == 0;
  }
};


struct greater_than
{
  __device__
  bool operator()(int x)
  {
    return x > device_removed_move_list_end;
  }
};


__global__ void printTempList(int temp_device_list[][PESIZE*FACTOR], int random_list_counter[PENUMBER]){

	for(int i =0; i<device_penumber_inuse; i++){
		for(int j=0; j<random_list_counter[i];j++){

			printf("%d ",temp_device_list[i][j]);

		}
		 printf("\n");
	}


}





__global__ void printList(int * list,int *removed_list_end){


	printf( "SIZE %d \n",removed_list_end - list) ;

	for(int i=0; i<removed_list_end - list; i++){
		printf("%d ",list[i]);
	
	}
	printf("\n");
}


__global__ void printListPre(int * list){


	printf( "SIZE %d \n",device_removed_space_list_end) ;

	for(int i=0; i<device_removed_space_list_end; i++){
		printf("%d ",list[i]);
	
	}
	printf("\n");
}


__global__ void prepareNewGrid (int new_grid[][SIZE+2], int * move_list, int permutation[][PESIZE]){
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	if(idx<device_removed_move_list_end){
		int idxTox = idx / PESIZE;
		int idxToy = idx % PESIZE;
		int agent_position = permutation[idxTox][idxToy];

		new_grid[agent_position/(SIZE+2)][agent_position%(SIZE+2)] = 0;

	}
}


__global__ void assign (int grid[][SIZE+2], int new_grid[][SIZE+2], int permutation[][PESIZE],  int * move_list, int * space_list, int samples[SAM_NUM_VALUES]){


	int idx=blockIdx.x*blockDim.x+threadIdx.x;

	if(idx < (device_removed_move_list_end) ){
		int idxTox = idx / PESIZE;
		int idxToy = idx % PESIZE;
		
		int space_position = space_list[samples[idx]-1];
		int agent_position = permutation[idxTox][idxToy];

		new_grid[space_position/(SIZE+2)][space_position%(SIZE+2)] = grid[agent_position/(SIZE+2)][agent_position%(SIZE+2)];  

	}


	    

}

__global__ void checkNumberDevice(int new_grid[][SIZE+2]){

	int agentTypeOne = 0;
	int agentTypeTwo = 0;


	for(int i=0; i<SIZE+2; i++){
		for(int j=0; j<SIZE+2; j++){
			if(new_grid[i][j] == 1){
				agentTypeOne +=1;	

			}
			else if(new_grid[i][j] == 2){
				agentTypeTwo += 1;

			}
		}

	}

	printf("Type One %d, Type Two %d\n",agentTypeOne, agentTypeTwo);




}

void checkNumber(int grid [SIZE+2][SIZE+2]){

	int agentTypeOne = 0;
	int agentTypeTwo = 0;


	for(int i=0; i<SIZE+2; i++){
		for(int j=0; j<SIZE+2; j++){
			if(grid[i][j] == 1){
				agentTypeOne +=1;	

			}
			else if(grid[i][j] == 2){
				agentTypeTwo += 1;

			}
		}

	}

	printf("Type One %d, Type Two %d\n",agentTypeOne, agentTypeTwo);




}


__global__ void devicePrintOutput(int device_list[][PESIZE]){

	for(int i =0; i<device_penumber_inuse; i++){

		//for(int j=0; j<random_list_counter[i];j++){
		// printf("%d \n",i);
		for(int j=0; j<PESIZE;j++){
			
			//printf("PE %d, index %d, value %d\n", i, j, device_list[i][j]);
			printf("%d ",device_list[i][j]);

		}
		printf("\n");
	}



}


__global__ void  initSamValue(int device_list[SAM_NUM_VALUES]){

	
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	device_list[idx] = idx+1;


}


__global__ void printSamples(int samples[SAM_NUM_VALUES]){

	
	for(int i=0; i<(device_removed_move_list_end); i++){

			printf("%d %d \n",i,samples[i]);
			
	}


}


__global__ void printSamValue(int device_sam_list[SAM_NUM_VALUES]){

	

	for(int i=0; i<(device_pe_inuse*SAM_PESIZE); i++){

			printf("%d ",device_sam_list[i]);
	}


}



int host_grid[SIZE+2][SIZE+2]; 

int main(int argc, char* argv[])
{

 	struct timespec start, stop;
    	double accum;
		
	
	int (*device_grid)[SIZE + 2];
	int (*device_newGrid)[SIZE + 2];

	int (*device_permutation_list)[PESIZE];
	int (*device_temp_permutation_list)[PESIZE*FACTOR];

	int (*random_list_counter);
	int (*scanned_random_list_counter);
	int (*move_list);
	int (*removed_move_list_end);

	int (*space_list); 
	int (*removed_space_list_end);

	int (*samples); 
	int (*device_sam_list);
	srand(SRAND_VALUE);

	size_t bytes = sizeof(int)*(SIZE + 2)*(SIZE + 2);

	myCurandState_t (*devState)[PESIZE];
	myCurandState_t (*devStateHyper);
	myCurandState_t (*devStateSam);

	cudaMalloc((void**)&devState, TOTAL * sizeof(myCurandState_t));

	cudaMalloc(&random_list_counter, sizeof(int)*(PENUMBER));

	cudaMalloc(&scanned_random_list_counter, sizeof(int)*(PENUMBER));

	cudaMalloc(&device_sam_list, sizeof(int)*(SAM_PESIZE)*(SAM_PENUMBER));

	cudaMalloc((void**)&device_grid, bytes);

	cudaMalloc((void**)&device_newGrid, bytes);

	cudaMalloc((void**)&device_permutation_list, sizeof(int)*(TOTAL));

	cudaMalloc((void**)&device_temp_permutation_list, sizeof(int)*(agentNumber)*FACTOR);

	cudaMalloc(&move_list, sizeof(int)*(SIZE + 2)*(SIZE + 2));

	cudaMalloc(&space_list, sizeof(int)*(SIZE + 2)*(SIZE + 2));

	cudaMalloc(&samples, sizeof(int)*(SAM_PESIZE)*(SAM_PENUMBER));

	cudaMalloc(&devStateHyper, SAM_PENUMBER * sizeof(myCurandState_t));

	cudaMalloc(&devStateSam, SAM_PENUMBER * sizeof(myCurandState_t));
	#ifdef DEBUG
          	cudaDeviceSynchronize();
		cudaCheckError();
   	#endif

	int blockSizeVerPermu = numThreadsPerBlock / PESIZE;
	dim3 blockSizePermu(blockSizeVerPermu, PESIZE, 1);

	initCurand<<<(ceil(TOTAL/double(numThreadsPerBlock))),blockSizePermu>>>(devState);

	#ifdef DEBUG
          	cudaDeviceSynchronize();
		cudaCheckError();
   	#endif

	for (int i=0; i<(SIZE+2); i++){
		for (int j=0; j<SIZE+2; j++){
			host_grid[i][j] = 0;
		}
	}



	int blockSizePerDim = sqrt(numThreadsPerBlock);
	int gridSizePerDim = (SIZE + 2) / blockSizePerDim;

	dim3 blockSize(blockSizePerDim, blockSizePerDim, 1);
	dim3 gridSize(gridSizePerDim, gridSizePerDim, 1);


	initPos(host_grid);
	//printOutput(host_grid);
   	#ifdef DEBUG
        cudaDeviceSynchronize();
		cudaCheckError();
   	#endif

	cudaMemcpy(device_grid,host_grid,bytes,cudaMemcpyHostToDevice);
	cudaMemcpy(device_newGrid,host_grid,bytes,cudaMemcpyHostToDevice);

	#ifdef DEBUG
    	cudaDeviceSynchronize();
		cudaCheckError();
	#endif


	initSamCurand<<<((double)SAM_PENUMBER / SAM_numThreadsPerBlock),SAM_numThreadsPerBlock>>>(devStateSam);

	#ifdef DEBUG
		cudaDeviceSynchronize();
		cudaCheckError();
	#endif

	update << <gridSize, blockSize >> >(device_grid, device_newGrid,move_list,space_list);
   	if( clock_gettime( CLOCK_REALTIME, &start) == -1 ) {
    	   perror( "clock gettime" );
   	   exit( EXIT_FAILURE );
   	 }


	cached_allocator alloc;
	int removed_list_number = 0;
	int space_list_number = 0;
	for(int i=0; i<ITERATIONS; i++){
   		 #ifdef DEBUG
          		cudaDeviceSynchronize();
			cudaCheckError();
   		 #endif

		compute << <gridSize, blockSize >> >(device_grid, device_newGrid, move_list, space_list, i);

   		 #ifdef DEBUG
          		cudaDeviceSynchronize();
			cudaCheckError();
   		 #endif




		removed_move_list_end = thrust::remove(thrust::cuda::par(alloc), move_list, move_list + ((SIZE+2)*(SIZE+2)), 0);


   		 #ifdef DEBUG
          		cudaDeviceSynchronize();
			cudaCheckError();
   		 #endif

		removed_list_number =  removed_move_list_end - move_list;

		cudaMemcpyToSymbol(device_removed_move_list_end, &removed_list_number, sizeof(int));


		int TwoDimGridSize = ceil(removed_list_number/double(numThreadsPerBlock));


   		 #ifdef DEBUG
          		cudaDeviceSynchronize();
			cudaCheckError();
   		 #endif
		removed_space_list_end = thrust::remove(thrust::cuda::par(alloc), space_list, space_list + ((SIZE+2)*(SIZE+2)), 0);
		
		space_list_number = removed_space_list_end - space_list;
   		 #ifdef DEBUG
          		cudaDeviceSynchronize();
			cudaCheckError();
   		 #endif


		cudaMemcpyToSymbol(device_removed_space_list_end, &space_list_number, sizeof(int));



		int penumberinuse = ceil(removed_list_number/ double(PESIZE));
		
		cudaMemcpyToSymbol(device_penumber_inuse, &penumberinuse, sizeof(int));

		generateList<<<ceil(space_list_number/double(numThreadsPerBlock)),blockSizePermu>>>(device_permutation_list);


        	int sam_num_inuse =  space_list_number;
		int sam_pe_inuse = ceil(double(sam_num_inuse) / SAM_PESIZE);	
		cudaMemcpyToSymbol(device_pe_inuse, &sam_pe_inuse, sizeof(int));
		cudaMemcpyToSymbol(device_num_inuse, &sam_num_inuse, sizeof(int));

		clearSamples<<<ceil(sam_pe_inuse*SAM_PESIZE / (double)SAM_numThreadsPerBlock), SAM_numThreadsPerBlock>>>(samples);

		#ifdef DEBUG
				cudaDeviceSynchronize();
				cudaCheckError();
		#endif

		int sam_gridSize = ceil((double)sam_pe_inuse / SAM_numThreadsPerBlock);

		initSamValue<<<ceil(double(sam_num_inuse) / SAM_numThreadsPerBlock), SAM_numThreadsPerBlock>>>(device_sam_list);



		#ifdef DEBUG
				cudaDeviceSynchronize();
				cudaCheckError();
		#endif


		sampleP<<<sam_gridSize, SAM_numThreadsPerBlock>>>( devStateSam, devStateHyper,  device_sam_list, samples, removed_list_number, 0, sam_pe_inuse-1);


		#ifdef DEBUG
 			cudaDeviceSynchronize();
 			cudaCheckError();
		#endif


		int OneDimGridSize = ceil(penumberinuse / double(numThreadsPerBlock));

		clearCounter<<<OneDimGridSize,(numThreadsPerBlock)>>>(random_list_counter);



   		 #ifdef DEBUG
          		cudaDeviceSynchronize();
			cudaCheckError();
   		 #endif


		sendToRandom<<<ceil(removed_list_number/double(numThreadsPerBlock)),blockSizePermu >>>(devState,move_list,device_temp_permutation_list,random_list_counter);



   		 #ifdef DEBUG
          		cudaDeviceSynchronize();
			cudaCheckError();
   		 #endif

		sortList<<<OneDimGridSize,(numThreadsPerBlock)>>>(device_temp_permutation_list,random_list_counter);


	   	 #ifdef DEBUG
          		cudaDeviceSynchronize();
			cudaCheckError();
   		 #endif	

		thrust::exclusive_scan(thrust::cuda::par(alloc), random_list_counter, random_list_counter + penumberinuse, scanned_random_list_counter); 


		randomPermute<<<OneDimGridSize,(numThreadsPerBlock)>>>(devState,device_temp_permutation_list,random_list_counter);
		
	   	 #ifdef DEBUG
          		cudaDeviceSynchronize();
			cudaCheckError();
   		 #endif	
		



	   	 #ifdef DEBUG
          		cudaDeviceSynchronize();
			cudaCheckError();
   		 #endif	
 		recoverSize<<<OneDimGridSize,(numThreadsPerBlock)>>>(device_permutation_list, device_temp_permutation_list,random_list_counter,scanned_random_list_counter);

		

		thrust::remove(thrust::device, samples, samples + sam_pe_inuse*SAM_PESIZE , 0);

	   	 #ifdef DEBUG
          		cudaDeviceSynchronize();
			cudaCheckError();
   		 #endif	

		prepareNewGrid <<<TwoDimGridSize, numThreadsPerBlock >>> (device_newGrid, move_list,device_permutation_list);
	   	 #ifdef DEBUG
          		cudaDeviceSynchronize();
			cudaCheckError();
   		 #endif	

		assign <<<TwoDimGridSize, numThreadsPerBlock>>> (device_grid, device_newGrid, device_permutation_list, move_list, space_list,samples);

   		 #ifdef DEBUG
          		cudaDeviceSynchronize();
			cudaCheckError();
   		 #endif

		update << <gridSize, blockSize >> >(device_grid, device_newGrid,move_list,space_list);
   		 #ifdef DEBUG
          		cudaDeviceSynchronize();
			cudaCheckError();
   		 #endif

	}


	if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) {
    	   perror( "clock gettime" );
   	   exit( EXIT_FAILURE );
   	 }

	accum = ( stop.tv_sec - start.tv_sec ) * 1e6
          + ( stop.tv_nsec - start.tv_nsec ) / 1e3;
	
    	printf( "%.1f Time is %.5f s \n",float(OCCUPANCY), accum / 1e6);
	cudaMemcpy(host_grid, device_newGrid, bytes, cudaMemcpyDeviceToHost);
	//printOutput(host_grid);
	//checkNumber(host_grid);
	
	cudaFree(device_grid);
	cudaFree(device_newGrid);
	cudaFree(device_permutation_list);
	cudaFree(device_temp_permutation_list);
	cudaFree(move_list);
	cudaFree(random_list_counter);
	cudaFree(scanned_random_list_counter);
	cudaFree(space_list);
	cudaFree(devState);
	cudaFree(samples);
	cudaFree(devStateSam);
	cudaFree(devStateHyper);
	cudaFree(device_sam_list);
	return 0;






}



void printOutput(int grid [SIZE+2][SIZE+2]  ){ //output grid from 1 t o SIZE+1
 	
	for (int i=1; i<SIZE+1; i++){
		for (int j=1; j<SIZE+1; j++){
			printf("%d ",grid[i][j]);
		//if(i%SIZE)
		}		
		printf("\n");
	}
	printf("\n");
}



void initPos(int grid [SIZE+2][SIZE+2]){  // type 1 and 2 to grid randomly
	int row;
	int column;
	for(int i=0; i<agentTypeOneNumber; i++){
		do{
			row = random_location();
			column = random_location();
		}while(grid[row][column] != 0);
		
		grid[row][column] = 1;	
	}

	for(int i=0; i<agentTypeTwoNumber; i++){
		do{
			row = random_location();
			column = random_location();
		}while(grid[row][column] != 0);
		
		grid[row][column] = 2;	
	}




}


int random_location() { //generate a random number from 1 to SIZE+1

	int r;

	r = rand();

	return (r % (SIZE) +1 );


}

