#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <time.h>
#include <unistd.h>
#include "parameter.cuh"


typedef curandStatePhilox4_32_10_t myCurandState_t; 


//#define DEBUG


#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   exit(0); \
 }                                                                 \
}

#define TOTAL (SIZE * SIZE)

#define SRAND_VALUE 200

#define CONFLICT_LIST_LENGTH 20

const int agentTypeOneNumber = agentNumber / 2;
const int agentTypeTwoNumber = agentNumber - agentTypeOneNumber;
const int happinessThreshold = 5;
const int numThreadsPerBlock = 256;

void printOutput(int [SIZE+2][SIZE+2]);
void initPos(int grid [SIZE+2][SIZE+2]);
int random_location();


__device__ unsigned int numberConflict = 0;
__device__ unsigned int numberMoveable = 0;


__device__ int getnextrand(myCurandState_t *state){

	int number = (1 + (int)(curand_uniform(state)*(SIZE)));
	return number;
}

__global__ void initCurand(myCurandState_t state[][SIZE+2], unsigned long seed){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int idy=blockIdx.y*blockDim.y+threadIdx.y;
	curand_init( 0 ,idx*(SIZE+2)+idy+10, 0, &state[idx][idy]);

}



__global__ void compute(int grid[][SIZE+2], int new_grid[][SIZE+2], int temp_grid[][SIZE+2]){
	
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	int idy=blockIdx.y*blockDim.y+threadIdx.y;
	int sameTypeCount=0;
	int current_priority = idx*(SIZE+2)+idy;


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

			temp_grid[idx][idy] = current_priority;
	}
	

}



__global__ void prepareNewGrid (int temp_grid[][SIZE+2], int new_grid[][SIZE+2], int new_gridCopy[][SIZE+2]){
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	int idy=blockIdx.y*blockDim.y+threadIdx.y;

	if(temp_grid[idx][idy] != 0){

		new_grid[idx][idy] = 0;

	}
	new_gridCopy[idx][idy] = new_grid[idx][idy];
}




__device__ bool agentsLeft;

__global__ void assign_ (myCurandState_t state[][SIZE+2],int grid[][SIZE+2], int new_grid[][SIZE+2], int temp_grid[][SIZE+2],int move_grid[][SIZE+2], int rowAndColumn[][SIZE+2]){

    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    int idy=blockIdx.y*blockDim.y+threadIdx.y;


    if(temp_grid[idx][idy] != 0 ){

        do {

		int row = getnextrand(&state[idx][idy]);
		int column = getnextrand(&state[idx][idy]);


             

		if(row>=1 && row <=SIZE && column>=1 && column<=SIZE && new_grid[row][column] == 0){
                	move_grid[idx][idy] = row * (SIZE + 2) + column;

                	return;

			}

		} while(true);
	}
}


__global__ void updateTonew (int grid[][SIZE+2], int new_grid[][SIZE+2], int new_gridCopy[][SIZE+2], int temp_grid[][SIZE+2],int move_grid[][SIZE+2], int rowAndColumn[][SIZE+2], myCurandState_t state[][SIZE + 2]){


    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    int idy=blockIdx.y*blockDim.y+threadIdx.y;

    int candidates[CONFLICT_LIST_LENGTH];
    int num_candidates = 0;

    int cell_id = idx * (SIZE + 2) + idy;
    if(!idx || !idy || idx > SIZE || idy > SIZE || new_gridCopy[idx][idy] != 0)
        return;
  
 
	
    for(int dx = 1; dx <= SIZE; dx++) {
        for(int dy = 1; dy <= SIZE; dy++) {

            int val = move_grid[dx][dy];
            if(val == cell_id) {
                candidates[num_candidates] = (dx) * (SIZE + 2) + (dy);
                num_candidates++;
                if(num_candidates >= CONFLICT_LIST_LENGTH)
                    printf("BUG! conflict list length exceeded\n");
            }
        }
    }


    if(!num_candidates)
        return;

    int priority = 0;
    if(num_candidates == 1)
        priority = candidates[0];

    if(num_candidates > 1){
        int r = getnextrand(&state[idx][idy]) % num_candidates;
        priority = candidates[r];
        agentsLeft = true;
    }

    int source_row = priority / (SIZE + 2);
    int source_col = priority % (SIZE + 2);

    new_grid[idx][idy] = grid[source_row][source_col];
    temp_grid[source_row][source_col] = 0;
}

__global__ void newTogrid (int grid[][SIZE+2], int new_grid[][SIZE+2]){

	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	int idy=blockIdx.y*blockDim.y+threadIdx.y;
	grid[idx][idy] = new_grid[idx][idy];
	

}

__global__ void clearMoveGrid (int move_grid[][SIZE+2]){

	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	int idy=blockIdx.y*blockDim.y+threadIdx.y;
	move_grid[idx][idy] = 0;

}

__global__ void update ( int temp_grid[][SIZE+2],int move_grid[][SIZE+2]){

	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	int idy=blockIdx.y*blockDim.y+threadIdx.y;
	temp_grid[idx][idy] = 0;
	move_grid[idx][idy] = 0;

}



__global__ void printConflict(){

	printf("Number of conflict %u\n", numberConflict);
	printf("Number of moveable %u\n", numberMoveable);

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
	printf("Supposed to be: Type One %d, Type Two %d\n", agentTypeOneNumber, agentTypeTwoNumber);




}


int host_grid[SIZE+2][SIZE+2]; 

int main(int argc, char* argv[])
{

	cudaDeviceSetLimit(cudaLimitPrintfFifoSize,  10*1024*1024);


 	struct timespec start, stop;
    	double accum;
	int (*device_grid)[SIZE + 2];
	int (*device_newGrid)[SIZE + 2];
        int (*device_newGridCopy)[SIZE + 2];

	int (*device_moveGrid)[SIZE + 2];

	int (*device_tempGrid)[SIZE + 2];

 	int (*device_rowAndColumn)[SIZE + 2];

	srand(SRAND_VALUE);

	size_t bytes = sizeof(int)*(SIZE + 2)*(SIZE + 2);
	myCurandState_t (*devState)[SIZE + 2];
	bool agentsRemain = false;
	
	cudaMalloc((void**)&devState, (SIZE+2)*(SIZE+2) * sizeof(myCurandState_t));

	cudaMalloc((void**)&device_grid, bytes);
	cudaMalloc((void**)&device_newGrid, bytes);
	cudaMalloc((void**)&device_newGridCopy, bytes);

	cudaMalloc((void**)&device_tempGrid, bytes);
	cudaMalloc((void**)&device_moveGrid, bytes);

	cudaMalloc((void**)&device_rowAndColumn, bytes);




	int blockSizePerDim = sqrt(numThreadsPerBlock);
	int gridSizePerDim = (SIZE + 2) / blockSizePerDim;

	dim3 blockSize(blockSizePerDim, blockSizePerDim, 1);
	dim3 gridSize(gridSizePerDim, gridSizePerDim, 1);

	initCurand<<<gridSize , blockSize>>>(devState, 1);
	for (int i=0; i<(SIZE+2); i++){
		for (int j=0; j<SIZE+2; j++){
			host_grid[i][j] = 0;
		}
	}


	
	initPos(host_grid);
//	printOutput(host_grid);

	cudaMemcpy(device_grid,host_grid,bytes,cudaMemcpyHostToDevice);
	cudaMemcpy(device_newGrid,host_grid,bytes,cudaMemcpyHostToDevice);
	


	newTogrid << <gridSize, blockSize >> >(device_grid, device_newGrid);

	update << <gridSize, blockSize >> >(device_tempGrid,device_moveGrid);
	if( clock_gettime( CLOCK_REALTIME, &start) == -1 ) {
    	   perror( "clock gettime" );
   	   exit( EXIT_FAILURE );
   	 }
	
	int numRoundsTotal = atoi(argv[1]);
	int roundCounter = 0;
    fprintf(stderr, "before loop\n");
	for(int i=0; i<numRoundsTotal; i++){

		roundCounter = 0;

		compute << <gridSize, blockSize >> >(device_grid, device_newGrid,device_tempGrid);

		 #ifdef DEBUG
			cudaDeviceSynchronize();
			cudaCheckError();
		 #endif
        fprintf(stderr, "before prepareNewGrid\n");
		
		
		prepareNewGrid<<<gridSize, blockSize>>>(device_tempGrid,device_newGrid,device_newGridCopy);


		 #ifdef DEBUG
			cudaDeviceSynchronize();
			cudaCheckError();
		 #endif


		do{

			agentsRemain = false;
			cudaMemcpyToSymbol(agentsLeft,&agentsRemain,sizeof(bool),0,cudaMemcpyHostToDevice);
			cudaDeviceSynchronize();
			cudaCheckError();

			assign_ << <gridSize, blockSize >> >(devState,device_grid, device_newGrid,device_tempGrid,device_moveGrid, device_rowAndColumn);
			updateTonew << <gridSize, blockSize >> >(device_grid, device_newGrid, device_newGridCopy, device_tempGrid,device_moveGrid, device_rowAndColumn, devState);
			clearMoveGrid<<<gridSize, blockSize >>>(device_moveGrid);
			roundCounter ++;
			cudaMemcpyFromSymbol(&agentsRemain,agentsLeft,sizeof(bool),0, cudaMemcpyDeviceToHost);

		}while(agentsRemain == true);


		newTogrid << <gridSize, blockSize >> >(device_grid, device_newGrid);

	
		update << <gridSize, blockSize >> >(device_tempGrid,device_moveGrid);
		

	}
	cudaDeviceSynchronize();


	
	if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) {
    	   perror( "clock gettime" );
   	   exit( EXIT_FAILURE );
   	 }

	accum = ( stop.tv_sec - start.tv_sec ) * 1e6
          + ( stop.tv_nsec - start.tv_nsec ) / 1e3;
	
    	printf( "Time is %.5f s \n", accum / 1e6);
	cudaMemcpy(host_grid, device_grid, bytes, cudaMemcpyDeviceToHost);
	//printOutput(host_grid);
	//checkNumber(host_grid);
	cudaFree(device_grid);
	cudaFree(device_newGrid);
        cudaFree(device_newGridCopy);
	cudaFree(device_tempGrid);
	cudaFree(devState);
	cudaFree(device_rowAndColumn);

	return 0;


}



void printOutput(int grid [SIZE+2][SIZE+2]  ){ //output grid from 1 t o SIZE+1
 	
	for (int i=0; i<SIZE+2; i++){
		for (int j=0; j<SIZE+2; j++){
			printf("%d ",grid[i][j]);
		//if(i%SIZE)
		}		
		printf("\n");
	}
	printf("\n");
}



void initPos(int grid [SIZE+2][SIZE+2]){  //assign type 1 and 2 to grid randomly
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


int random_location() { //generate a random number from 1 to SIZE

	int r;

	r = rand();

	return (r % (SIZE) +1 );


}










