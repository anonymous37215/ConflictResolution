#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <time.h>
#include "parameter.cuh"
#include <fstream>


typedef curandStatePhilox4_32_10_t myCurandState_t; 

#define DEBUG
#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   exit(0); \
 }                                                                 \
}
#define TOTAL (SIZE * SIZE)
#define SRAND_VALUE 200

using namespace std;

//const int agentNumber = (SIZE * SIZE / 2);


const int agentTypeOneNumber = agentNumber / 2;
const int agentTypeTwoNumber = agentNumber - agentTypeOneNumber;
const int happinessThreshold = 5;
const int numThreadsPerBlock = 256;

void printOutput(int[SIZE + 2][SIZE + 2]);
void printToFile(int[SIZE + 2][SIZE + 2]);
void initPos(int grid[SIZE + 2][SIZE + 2]);
int random_location();

__device__ unsigned int numberConflict = 0;
__device__ unsigned int numberMoveable = 0;

__device__ int getnextrand(myCurandState_t *state){

	
	return (1 + (int)(curand_uniform(state)*(SIZE)));
}

__global__ void initCurand(myCurandState_t state[][SIZE + 2],
						   unsigned long seed)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	curand_init(idx * (SIZE + 2) + idy, 0, 0, &state[idx][idy]);
}



__global__ void compute(int grid[][SIZE + 2], int new_grid[][SIZE + 2],
						int temp_grid[][SIZE + 2],
						int move_grid[][SIZE + 2])
{

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int sameTypeCount = 0;
	int current_priority = idx * (SIZE + 2) + idy;

	if (grid[idx][idy] != 0) {
		int currentType = grid[idx][idy];

		if (grid[idx - 1][idy - 1] == currentType) {
			sameTypeCount += 1;
		}

		if (grid[idx - 1][idy] == currentType) {
			sameTypeCount += 1;
		}

		if (grid[idx - 1][idy + 1] == currentType) {
			sameTypeCount += 1;
		}

		if (grid[idx][idy - 1] == currentType) {
			sameTypeCount += 1;
		}

		if (grid[idx][idy + 1] == currentType) {
			sameTypeCount += 1;
		}

		if (grid[idx + 1][idy - 1] == currentType) {
			sameTypeCount += 1;
		}

		if (grid[idx + 1][idy] == currentType) {
			sameTypeCount += 1;
		}

		if (grid[idx + 1][idy + 1] == currentType) {
			sameTypeCount += 1;
		}

		if (sameTypeCount < happinessThreshold) {



			temp_grid[idx][idy] = current_priority;

		}
	}


}

__device__ bool agentsLeft;

__global__ void assign(myCurandState_t state[][SIZE + 2], int grid[][SIZE + 2],
					   int new_grid[][SIZE + 2], int temp_grid[][SIZE + 2],
					   int move_grid[][SIZE + 2],
					   int rowAndColumn[][SIZE + 2])
{

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;

	int current_priority = idx * (SIZE + 2) + idy;
	int row = 0;
	int column = 0;
	int new_value;
	bool success = false;
	if (temp_grid[idx][idy] != 0) {

		do {
			row = getnextrand(&state[idx][idy]);
			column = getnextrand(&state[idx][idy]);

			if (new_grid[row][column] == 0) {


				new_value =
					atomicMax(&move_grid[row][column], current_priority);

				if (new_value == 0) {	//find an empty cell

					success = true;

				}



				else if (new_value < current_priority) {	//agent with lower priority inside the cell

					atomicAdd(&numberConflict, 1);
					idx = new_value / (SIZE + 2);
					idy = new_value % (SIZE + 2);
					current_priority = new_value;

				}

				else{  //agent with higher priority inside the cell -> repeat

				atomicAdd(&numberConflict, 1);

				}


			}
		} while (!success);
	}
}


__global__ void updateTonew(int grid[][SIZE + 2], int new_grid[][SIZE + 2],
							int temp_grid[][SIZE + 2],
							int move_grid[][SIZE + 2],
							int rowAndColumn[][SIZE + 2])
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;

	int row = idx;
	int column = idy;

	int current_priority = move_grid[row][column];

	if (idx + idy != 0 && current_priority) {

		int source_row = current_priority / (SIZE + 2);
		int source_col = current_priority % (SIZE + 2);
		new_grid[row][column] = grid[source_row][source_col];

		temp_grid[source_row][source_col] = 0;

	}
}





__global__ void prepareNewGrid(int temp_grid[][SIZE + 2],
							   int new_grid[][SIZE + 2])
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;

	if (temp_grid[idx][idy] != 0) {
		new_grid[idx][idy] = 0;
	}
}





__global__ void newTogrid(int grid[][SIZE + 2], int new_grid[][SIZE + 2])
{

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	grid[idx][idy] = new_grid[idx][idy];

}



__global__ void update(int temp_grid[][SIZE + 2],
					   int move_grid[][SIZE + 2],
					   int rowAndColumn[][SIZE + 2])
{

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;

	temp_grid[idx][idy] = 0;
	move_grid[idx][idy] = 0;


}




void checkNumber(int grid[SIZE + 2][SIZE + 2])
{

	int agentTypeOne = 0;
	int agentTypeTwo = 0;


	for (int i = 0; i < SIZE + 2; i++) {
		for (int j = 0; j < SIZE + 2; j++) {
			if (grid[i][j] == 1) {
				agentTypeOne += 1;

			} else if (grid[i][j] == 2) {
				agentTypeTwo += 1;

			}
		}

	}

	printf("Type One %d, Type Two %d\n", agentTypeOne, agentTypeTwo);




}




int host_grid[SIZE + 2][SIZE + 2];

int main(int argc, char *argv[])
{
	cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1024 * 1024 * 100);
	struct timespec start, stop;
	double accum;
	int (*device_grid)[SIZE + 2];
	int (*device_newGrid)[SIZE + 2];
	int (*device_moveGrid)[SIZE + 2];

	int (*device_tempGrid)[SIZE + 2];

	int (*device_rowAndColumn)[SIZE + 2];

	srand(SRAND_VALUE);

	size_t bytes = sizeof(int) * (SIZE + 2) * (SIZE + 2);
	myCurandState_t(*devState)[SIZE + 2];

	cudaMalloc((void **) &devState,
			   (SIZE + 2) * (SIZE + 2) * sizeof(myCurandState_t));

	cudaMalloc((void **) &device_grid, bytes);
	cudaMalloc((void **) &device_newGrid, bytes);
	cudaMalloc((void **) &device_tempGrid, bytes);
	cudaMalloc((void **) &device_moveGrid, bytes);

	cudaMalloc((void **) &device_rowAndColumn, bytes);



	int blockSizePerDim = sqrt(numThreadsPerBlock);
	int gridSizePerDim = (SIZE + 2) / blockSizePerDim;

	dim3 blockSize(blockSizePerDim, blockSizePerDim, 1);
	dim3 gridSize(gridSizePerDim, gridSizePerDim, 1);

	initCurand <<< gridSize, blockSize >>> (devState, 1);
	for (int i = 0; i < (SIZE + 2); i++) {
		for (int j = 0; j < SIZE + 2; j++) {
			host_grid[i][j] = 0;
		}
	}



	initPos(host_grid);
	// printOutput(host_grid);

	cudaMemcpy(device_grid, host_grid, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(device_newGrid, host_grid, bytes, cudaMemcpyHostToDevice);

	newTogrid << <gridSize, blockSize >> >(device_grid, device_newGrid);

	update << <gridSize, blockSize >> >(device_tempGrid, device_moveGrid,
										device_rowAndColumn);
	if (clock_gettime(CLOCK_REALTIME, &start) == -1) {
		perror("clock gettime");
		exit(EXIT_FAILURE);
	}

	int numRoundsTotal = atoi(argv[1]);
	for (int i = 0; i < numRoundsTotal; i++) {


		compute << <gridSize, blockSize >> >(device_grid, device_newGrid,
											 device_tempGrid,
											 device_moveGrid);

#ifdef DEBUG
		cudaDeviceSynchronize();
		cudaCheckError();
#endif

		prepareNewGrid <<< gridSize, blockSize >>> (device_tempGrid,
													device_newGrid);

		assign << <gridSize, blockSize >> >(devState, device_grid,
											device_newGrid,
											device_tempGrid,
											device_moveGrid,
											device_rowAndColumn);
#ifdef DEBUG
		cudaDeviceSynchronize();
		cudaCheckError();
#endif

		updateTonew << <gridSize, blockSize >> >(device_grid,
												 device_newGrid,
												 device_tempGrid,
												 device_moveGrid,
												 device_rowAndColumn);
#ifdef DEBUG
		cudaDeviceSynchronize();
		cudaCheckError();
#endif

#ifdef DEBUG
		cudaDeviceSynchronize();
		cudaCheckError();
#endif
		newTogrid << <gridSize, blockSize >> >(device_grid,
											   device_newGrid);
#ifdef DEBUG
		cudaDeviceSynchronize();
		cudaCheckError();
#endif


		update << <gridSize, blockSize >> >(device_tempGrid,
											device_moveGrid,
											device_rowAndColumn);



	}
	cudaDeviceSynchronize();


	if (clock_gettime(CLOCK_REALTIME, &stop) == -1) {
		perror("clock gettime");
		exit(EXIT_FAILURE);
	}

	accum = (stop.tv_sec - start.tv_sec) * 1e6
		+ (stop.tv_nsec - start.tv_nsec) / 1e3;

	printf( "%.1f Time is %.5f s \n",float(OCCUPANCY), accum / 1e6);
	cudaMemcpy(host_grid, device_grid, bytes, cudaMemcpyDeviceToHost);
	//printOutput(host_grid);
	//checkNumber(host_grid);
	cudaFree(device_grid);
	cudaFree(device_newGrid);
	cudaFree(device_tempGrid);
	cudaFree(devState);
	cudaFree(device_rowAndColumn);

	return 0;


}



void printOutput(int grid[SIZE + 2][SIZE + 2])
{								//output grid from 1 t o SIZE+1

	for (int i = 1; i < SIZE + 1; i++) {
		for (int j = 1; j < SIZE + 1; j++) {
			printf("%d ", grid[i][j]);
			//if(i%SIZE)
		}
		printf("\n");
	}
	printf("\n");
}



void initPos(int grid[SIZE + 2][SIZE + 2])
{								//assign type 1 and 2 to grid randomly
	int row;
	int column;
	for (int i = 0; i < agentTypeOneNumber; i++) {
		do {
			row = random_location();
			column = random_location();
		} while (grid[row][column] != 0);

		grid[row][column] = 1;
	}

	for (int i = 0; i < agentTypeTwoNumber; i++) {
		do {
			row = random_location();
			column = random_location();
		} while (grid[row][column] != 0);

		grid[row][column] = 2;
	}




}


int random_location()
{								//generate a random number from 1 to SIZE

	int r;

	r = rand();

	return (r % (SIZE) + 1);


}

void printToFile(int grid [SIZE+2][SIZE+2]  ){ //output grid from 1 t o SIZE+1
 	static int num;

        char outfilename[128];
        snprintf(outfilename, 128, "img/out%08d.pgm", num);
        num++;
      
        char header[128];
        snprintf(header, 128, "P6\n%d %d 255\n", SIZE, SIZE);

        std::ofstream resultfile(outfilename, std::ofstream::binary);
      
        if (resultfile) {
          resultfile.write(header, strlen(header));
      
          if (!resultfile) {
            fprintf(stderr, "Error: header could be written to result file.\n");
            exit(-1);
          }

          struct pixel { unsigned char r = 0, g = 0, b = 0; } col;
          for (int i=1; i<SIZE+1; i++){
            for (int j=1; j<SIZE+1; j++){
              col.r = col.g = col.b = 0;
              // printf("%d/%d is %d\n", i, j, grid[i][j]);
              if(grid[i][j] == 1) {
                col.r = 180;
              }
              if(grid[i][j] == 2) {
                col.b = 180;
              }
              // printf("trying to write %d/%d/%d\n", col.r, col.g, col.b);
              resultfile.write(reinterpret_cast<char *>(&col), 3 * sizeof(unsigned char));
            }
          }
          resultfile.close();
          system("for i in img/out*.pgm; do convert $i $i.png; done; rm img/*.pgm");
        }
        else {
          fprintf(stderr, "Could not open result file.\n");
          exit(-1);
        }

}


