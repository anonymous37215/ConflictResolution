#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <time.h>
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

#define CONFLICT_LIST_LENGTH 10

//const int agentNumber = (SIZE * SIZE / 2);
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
	//printf("%d\n",number);
	return number;
}

__global__ void initCurand(myCurandState_t state[][SIZE+2], unsigned long seed){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int idy=blockIdx.y*blockDim.y+threadIdx.y;
	//curand_init(seed, idx*(SIZE+2)+idy, 0, &state[idx][idy]);
	curand_init( 0 ,idx*(SIZE+2)+idy+10, 0, &state[idx][idy]);
	//curand_init(idx*(SIZE+2)+idy, 0, 0, &state[idx][idy]);

}



__global__ void compute(int grid[][SIZE+2], int new_grid[][SIZE+2], int temp_grid[][SIZE+2]){
	
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	int idy=blockIdx.y*blockDim.y+threadIdx.y;
	int sameTypeCount=0;
	int current_priority = idx*(SIZE+2)+idy;

	//new_grid[idx][idy] = grid[idx][idy];

	if(grid[idx][idy] != 0){
		//new_grid[idx][idy] = 3;
		//for(int i=-1;i<2;i++){
		//	for(int j=-1; j<2;j++){
		//		if(i!=0 && j!=0 && (grid[idx+i][idy+j]==grid[idx][idy]) ){
					
		//			sameTypeCount += 1;
		//		} 
		//	}
		//}
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

			// printf("moveable: %d\n", current_priority);
			
			temp_grid[idx][idy] =  current_priority;
       			//atomicAdd(&numberMoveable, 1);
		}
	}
	

}



__global__ void prepareNewGrid (int temp_grid[][SIZE+2], int new_grid[][SIZE+2]){
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	int idy=blockIdx.y*blockDim.y+threadIdx.y;

	if(temp_grid[idx][idy] != 0){
		//int agent_position = move_list[idx];
		//int idxTox = idx / PESIZE;
		//int idxToy = idx % PESIZE;
		//int agent_position = permutation[idxTox][idxToy];

		//if(agent_position/(SIZE+2)>(SIZE+2)){
		//	printf("Outside %d %d\n",idx,agent_position);
		//}
		//else{
		new_grid[idx][idy] = 0;

		//}
	}
}




__device__ bool agentsLeft;

__global__ void assign_ (myCurandState_t state[][SIZE+2],int grid[][SIZE+2], int new_grid[][SIZE+2], int temp_grid[][SIZE+2],int move_grid[][SIZE+2][CONFLICT_LIST_LENGTH], int move_grid_counters[][SIZE+2], int rowAndColumn[][SIZE+2]){

	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	int idy=blockIdx.y*blockDim.y+threadIdx.y;
	int current_priority = idx*(SIZE+2)+idy;	
	int row = 0;
	int column = 0;
	int old_value;
	int round =0;

	if(temp_grid[idx][idy] != 0 ){
			//if(!iteration)
			//if(current_priority == 2172)
        // printf("%d wants to move\n", current_priority);

		while(true) {
			row = getnextrand(&state[idx][idy]);
			column = getnextrand(&state[idx][idy]);
			if(new_grid[row][column] == 0){
                
				old_value = atomicAdd(&move_grid_counters[row][column], 1);
                // printf("%d %d going to %d %d, number in line: %d\n", idx, idy, row, column, old_value);
                move_grid[row][column][old_value] = current_priority;

				break;

			}
			round++;
		}
	}
}


__global__ void updateTonew (int grid[][SIZE+2], int new_grid[][SIZE+2],int temp_grid[][SIZE+2],int move_grid[][SIZE+2][CONFLICT_LIST_LENGTH], int move_grid_counters[][SIZE+2], int rowAndColumn[][SIZE+2], myCurandState_t state[][SIZE + 2]){
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	int idy=blockIdx.y*blockDim.y+threadIdx.y;
	
    int num_agents = move_grid_counters[idx][idy];

    if(!num_agents)
        return;

    int priority = 0;
    if(num_agents == 1)
        priority = move_grid[idx][idy][0];

    if(num_agents > 1){
        int r = getnextrand(&state[idx][idy]) % num_agents;
        priority = move_grid[idx][idy][r];
        agentsLeft = true;
    }

   	int source_row = priority / (SIZE + 2);
   	int source_col = priority % (SIZE + 2);
   	new_grid[idx][idy] = grid[source_row][source_col];
   	temp_grid[source_row][source_col] = 0;
}

__global__ void newTogrid (int grid[][SIZE+2], int new_grid[][SIZE+2]){

	//int (*device_tmpGrid)[SIZE+2]; 
	//device_tmpGrid = grid;
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	int idy=blockIdx.y*blockDim.y+threadIdx.y;
	grid[idx][idy] = new_grid[idx][idy];
	//temp_grid[idx][idy] = 0;
	//move_grid[idx][idy] = 0;
	//new_grid = device_tmpGrid;
	

}

__global__ void clearMoveGrid (int move_grid_counters[][SIZE+2]){

	//int (*device_tmpGrid)[SIZE+2];
	//device_tmpGrid = grid;
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	int idy=blockIdx.y*blockDim.y+threadIdx.y;
	//grid[idx][idy] = new_grid[idx][idy];
	//temp_grid[idx][idy] = 0;
	// move_grid[idx][idy] = 0;
    move_grid_counters[idx][idy] = 0;
	//new_grid = device_tmpGrid;


}

__global__ void update ( int temp_grid[][SIZE+2],int move_grid_counters[][SIZE+2]){

	//int (*device_tmpGrid)[SIZE+2]; 
	//device_tmpGrid = grid;
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	int idy=blockIdx.y*blockDim.y+threadIdx.y;
	//grid[idx][idy] = new_grid[idx][idy];
	temp_grid[idx][idy] = 0;
	move_grid_counters[idx][idy] = 0;
	//new_grid = device_tmpGrid;
	

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




}


int host_grid[SIZE+2][SIZE+2]; 

int main(int argc, char* argv[])
{

	cudaDeviceSetLimit(cudaLimitPrintfFifoSize,  10*1024*1024);


 	struct timespec start, stop;
    	double accum;
	int (*device_grid)[SIZE + 2];
	int (*device_newGrid)[SIZE + 2];
	int (*device_moveGrid)[SIZE + 2][CONFLICT_LIST_LENGTH];
	int (*device_moveGridCounters)[SIZE + 2];

	int (*device_tempGrid)[SIZE + 2];

 	int (*device_rowAndColumn)[SIZE + 2];

	//int (*device_tmpGrid)[SIZE+2]; 
	srand(SRAND_VALUE);

	size_t bytes = sizeof(int)*(SIZE + 2)*(SIZE + 2);
	//host_grid = (int*)malloc(bytes);
	myCurandState_t (*devState)[SIZE + 2];
	bool agentsRemain = false;
	//bool  *agentsRemain = new bool();
	//*agentsRemain = false;
	//bool * agentsLeft;
	//typeof(agentsLeft) agentsRemain;
	
	cudaMalloc((void**)&devState, (SIZE+2)*(SIZE+2) * sizeof(myCurandState_t));

	cudaMalloc((void**)&device_grid, bytes);
	cudaMalloc((void**)&device_newGrid, bytes);
	cudaMalloc((void**)&device_tempGrid, bytes);
	cudaMalloc((void**)&device_moveGrid, bytes * CONFLICT_LIST_LENGTH);
	cudaMalloc((void**)&device_moveGridCounters, bytes);

	cudaMalloc((void**)&device_rowAndColumn, bytes);

	//cudaMalloc(&agentsLeft, sizeof(bool));



	int blockSizePerDim = sqrt(numThreadsPerBlock);
	int gridSizePerDim = (SIZE + 2) / blockSizePerDim;

	dim3 blockSize(blockSizePerDim, blockSizePerDim, 1);
	dim3 gridSize(gridSizePerDim, gridSizePerDim, 1);

	initCurand<<<gridSize , blockSize>>>(devState, 1);
	//cudaDeviceSynchronize();
	for (int i=0; i<(SIZE+2); i++){
		for (int j=0; j<SIZE+2; j++){
			host_grid[i][j] = 0;
		}
	}


	
	initPos(host_grid);
	// printOutput(host_grid);

	cudaMemcpy(device_grid,host_grid,bytes,cudaMemcpyHostToDevice);
	cudaMemcpy(device_newGrid,host_grid,bytes,cudaMemcpyHostToDevice);
	//cudaMemcpy(device_newGrid,device_grid,bytes,cudaMemcpyDeviceToDevice);
	


	newTogrid << <gridSize, blockSize >> >(device_grid, device_newGrid);
	//cudaDeviceSynchronize();
	//cudaCheckError();

	update << <gridSize, blockSize >> >(device_tempGrid,device_moveGridCounters);
	//cudaDeviceSynchronize();
	//cudaCheckError();
	//clock_t begin = clock(); //timing
	if( clock_gettime( CLOCK_REALTIME, &start) == -1 ) {
    	   perror( "clock gettime" );
   	   exit( EXIT_FAILURE );
   	 }
	
	int numRoundsTotal = atoi(argv[1]);
	int roundCounter = 0;
    // fprintf(stderr, "before loop\n");
	for(int i=0; i<numRoundsTotal; i++){

		roundCounter = 0;

       // fprintf(stderr, "before compute\n");
		compute << <gridSize, blockSize >> >(device_grid, device_newGrid,device_tempGrid);
        //fprintf(stderr, "after compute\n");

		 #ifdef DEBUG
			cudaDeviceSynchronize();
			cudaCheckError();
		 #endif
        // fprintf(stderr, "before prepareNewGrid\n");
		
		//cudaMemcpy(host_grid, device_newGrid, bytes, cudaMemcpyDeviceToHost);

		
		prepareNewGrid<<<gridSize, blockSize>>>(device_tempGrid,device_newGrid);


        // fprintf(stderr, "before do loop\n");
		 #ifdef DEBUG
			cudaDeviceSynchronize();
			cudaCheckError();
		 #endif


		do{

			agentsRemain = false;
			cudaMemcpyToSymbol(agentsLeft,&agentsRemain,sizeof(bool),0,cudaMemcpyHostToDevice);

			assign_ << <gridSize, blockSize >> >(devState,device_grid, device_newGrid,device_tempGrid,device_moveGrid, device_moveGridCounters, device_rowAndColumn);
			// cudaDeviceSynchronize();
			// cudaCheckError();
			
			updateTonew << <gridSize, blockSize >> >(device_grid, device_newGrid,device_tempGrid,device_moveGrid,device_moveGridCounters,device_rowAndColumn, devState);
			//update agentLeft
			// cudaDeviceSynchronize();
			// cudaCheckError();

			//cudaMemcpy(agentsRemain, agentsLeft, sizeof(bool), cudaMemcpyDeviceToHost);
			//cudaCheckError();
			clearMoveGrid<<<gridSize, blockSize >>>(device_moveGridCounters);
			//newTogrid << <gridSize, blockSize >> >(device_grid, device_newGrid);
			// cudaDeviceSynchronize();
			// cudaCheckError();
			roundCounter ++;
			cudaMemcpyFromSymbol(&agentsRemain,agentsLeft,sizeof(bool),0, cudaMemcpyDeviceToHost);

		//}while(false && agentsRemain == true);
		}while(agentsRemain == true);
		// printf("roundCounter %d  numRounds %d\n", roundCounter,i);
		//while there are agents left

		// cudaDeviceSynchronize();
		//cudaCheckError();

		//newTogrid << <gridSize, blockSize >> >(device_grid, device_newGrid);

		newTogrid << <gridSize, blockSize >> >(device_grid, device_newGrid);

	
		update << <gridSize, blockSize >> >(device_tempGrid,device_moveGridCounters);
		
		//cudaMemcpy(device_grid,host_grid,bytes,cudaMemcpyHostToDevice);
		//cudaMemcpy(device_newGrid,host_grid,bytes,cudaMemcpyHostToDevice);

		


		//printOutput(host_grid);


	}
	cudaDeviceSynchronize();



	//clock_t end = clock();
	//double time_spent = (double)(end - begin)*1000.0 / CLOCKS_PER_SEC;
	//printf("Time: %f ms per round\n",time_spent / numRoundsTotal);
	
	if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) {
    	   perror( "clock gettime" );
   	   exit( EXIT_FAILURE );
   	 }

	accum = ( stop.tv_sec - start.tv_sec ) * 1e6
          + ( stop.tv_nsec - start.tv_nsec ) / 1e3;
	
    	printf( "%.1f Time is %.5f s \n",float(OCCUPANCY), accum / 1e6);
	//printConflict<<<1,1>>>();
	cudaMemcpy(host_grid, device_grid, bytes, cudaMemcpyDeviceToHost);
	//printOutput(host_grid);
	//checkNumber(host_grid);
	cudaFree(device_grid);
	cudaFree(device_newGrid);
	cudaFree(device_tempGrid);
	cudaFree(devState);
	//cudaFree(agentsLeft);
	cudaFree(device_rowAndColumn);
	//free(host_grid);

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










