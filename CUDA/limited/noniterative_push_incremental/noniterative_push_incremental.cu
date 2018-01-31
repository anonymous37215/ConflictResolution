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

// #define DEBUG

#define CEILING(x,y) (((x) + (y) - 1) / (y))

#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   exit(0); \
 }                                                                 \
}

#define FACTOR 10
#define ITERATIONS 100
#define TOTAL (SIZE * SIZE)

#define GRIDSIZE (SIZE+2)
#define GRIDTOTAL (SIZE+2)*(SIZE+2)
#define PENUMBER (CEILING(TOTAL, PESIZE))

int host_grid[SIZE + 2][SIZE + 2];

const int agentTypeOneNumber = agentNumber / 2;
const int agentTypeTwoNumber = agentNumber - agentTypeOneNumber;
const int happinessThreshold = 5;
const int limitedNeighbourhood = 3;

void printOutput(int[SIZE + 2][SIZE + 2]);
void initPos(int grid[SIZE + 2][SIZE + 2]);
int random_location();

void printOutput(int grid[SIZE + 2][SIZE + 2])
{								//output grid from 1 t o SIZE+1

	for (int i = 1; i < SIZE + 1; i++) {
        printf("XXX: ");
		for (int j = 1; j < SIZE + 1; j++) {
            if(grid[i][j])
  			  printf("%d ", grid[i][j]);
            else
  			  printf("  ");
		}
		printf("\n");
	}
	printf("\n");
}

void checkOutput(int grid[SIZE + 2][SIZE + 2])
{								//output grid from 1 t o SIZE+1

    double sum_x = 0, sum_y = 0;
    int num_type_one = 0, num_type_two = 0;
	for (int i = 1; i < SIZE + 1; i++) {
		for (int j = 1; j < SIZE + 1; j++) {
            if(grid[i][j] != 0) {
                sum_x += i;
                sum_y += j;
            }
            
            if(grid[i][j] == 1)
                num_type_one++;
            else if(grid[i][j] == 2)
                num_type_two++;
		}
	}
    printf("%d agents of type one, should be %d\n", num_type_one, agentTypeOneNumber);
    printf("%d agents of type two, should be %d\n", num_type_two, agentTypeTwoNumber);
    int num_agents = num_type_one + num_type_two;
    printf("avg position: %.5f %.5f\n", sum_x / num_agents / (SIZE + 2), sum_y / num_agents / (SIZE + 2));

    if(num_type_one != agentTypeOneNumber || num_type_two != agentTypeTwoNumber) {
        printf("BUG: we lost some agents\n");
        exit(1);
    }
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
{								//generate a random number from 1 to SIZE+1

	int r;

	r = rand();

	return (r % (SIZE) + 1);


}
__global__ void updateTonew(int grid[][SIZE + 2], int new_grid[][SIZE + 2],
                            int temp_grid[][SIZE + 2],
                            int move_grid[][SIZE + 2],
                            int rowAndColumn[][SIZE + 2],
                            int permutation_inverse[][SIZE])
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    int row = idx + 1; // TODO: very uncertain about this
    int column = idy + 1;

    int current_priority = move_grid[row][column];

    if (current_priority) {
        current_priority--;

        int index_row = current_priority / SIZE;
        int index_col = current_priority % SIZE;

        int source_identifier = permutation_inverse[index_row][index_col];

        int source_row = source_identifier % SIZE + 1;
        int source_col = source_identifier / SIZE + 1;

        new_grid[row][column] = grid[source_row][source_col];

        temp_grid[source_row][source_col] = 0;

    }
}


__global__ void newTogrid(int grid[][SIZE + 2], int new_grid[][SIZE + 2])
{

    int row = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int col = blockIdx.y * blockDim.y + threadIdx.y + 1;

    grid[row][col] = new_grid[row][col];
}

__device__ int getnextrandint(myCurandState_t *state){


    return (1 + (int)(curand_uniform(state)*(SIZE)));
}

__device__ float getnextrandfloat(myCurandState_t *state){


    return (curand_uniform(state));
}


__global__ void compute(int grid[][SIZE + 2], int new_grid[][SIZE + 2],
                        int temp_grid[][SIZE + 2],
                        int move_grid[][SIZE + 2],
                        int permutation[][PESIZE])
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


__global__ void assign(myCurandState_t *state, int grid[][SIZE + 2],
                       int new_grid[][SIZE + 2], int temp_grid[][SIZE + 2],
                       int move_grid[][SIZE + 2],
                       int rowAndColumn[][SIZE + 2], int permutation[][PESIZE], int permutation_inverse[][SIZE], int local_trials[][SIZE + 2])
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idx_linear = (idy - 1) * SIZE + (idx - 1);

    if(idx == 0 || idx > SIZE || idy == 0 || idy > SIZE)
        return;

    int current_priority = permutation[idx_linear / PESIZE][idx_linear % PESIZE] + 1;

    int row = 0;
    int column = 0;
    int old_value;

    bool success = false;

    int localLimitedNeighbourhood;

    if (temp_grid[idx][idy] != 0) {
        do {
            local_trials[idx][idy]++;

            localLimitedNeighbourhood = limitedNeighbourhood * ((local_trials[idx][idy] / 10) + 1);

            int randomRow = (getnextrandint(&state[idx * (SIZE + 2) + idy]) % localLimitedNeighbourhood) - (localLimitedNeighbourhood / 2);
            row = idx + randomRow;
 
            int randomColumn = (getnextrandint(&state[idx * (SIZE + 2) + idy]) % localLimitedNeighbourhood) - (localLimitedNeighbourhood / 2);
 
            column = idy + randomColumn;

            if (new_grid[row][column] == 0 && row>=1 && row <=SIZE && column>=1 && column<=SIZE) {

                old_value =
                    atomicMax(&move_grid[row][column], current_priority);

                if (old_value == 0) {   //find an empty cell
                    success = true;
                }



                else if (old_value < current_priority) {    //agent with lower priority inside the cell
                    int index_row = (old_value - 1) / SIZE;
                    int index_col = (old_value - 1) % SIZE;
            
                    int source_identifier = permutation_inverse[index_row][index_col];

                    idx = source_identifier % SIZE + 1;
                    idy = source_identifier / SIZE + 1;

                    current_priority = old_value;
                }

            }

        } while (!success);
    }
    //store the value of column and row

}

__global__ void prepareNewGrid(int temp_grid[][SIZE + 2],
                               int new_grid[][SIZE + 2],
                               int local_trials[][SIZE + 2])
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (temp_grid[idx][idy] != 0) {
        new_grid[idx][idy] = 0;
    }
    local_trials[idx][idy] = 0;
}


__global__ void update(int temp_grid[][SIZE + 2],
                       int move_grid[][SIZE + 2],
                       int rowAndColumn[][SIZE + 2])
{

    int row = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int col = blockIdx.y * blockDim.y + threadIdx.y + 1;

    temp_grid[row][col] = 0;
    move_grid[row][col] = 0;

}

__global__ void printPermutedListInverse(int a[][SIZE])
{
    for (int y = 0; y < SIZE; y++)
        for (int x = 0; x < SIZE; x++)
            printf("i: %d -> %d\n", a[y][x], y * SIZE + x);

    printf("\n\n");
}


__global__ void printPermutedList(int a[][PESIZE])
{
    printf("a\n");
    for (int pe = 0; pe < PENUMBER; pe++)
        for (int i = 0; i < PESIZE; i++)
            printf("f: %d -> %d\n", pe * PESIZE + i, a[pe][i]);

    printf("\n\n");
}

__global__ void printTempPermutedList(int a[][PESIZE * FACTOR], int n)
{
    for (int pe = 0; pe < PENUMBER; pe++)
        for (int i = 0; i < PESIZE * FACTOR; i++)
            printf("%d,", a[pe][i]);

    printf("\n\n");
}


__global__ void initCurand(myCurandState_t *state, int seed_offset)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < GRIDTOTAL) {
		curand_init(idx + seed_offset, 0, 0, &state[idx]);
	}
}

__global__ void initPermutationList(int device_list[][PESIZE])
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;

    	if(idx * PESIZE + idy < (SIZE * SIZE)) {
    		device_list[idx][idy] = idx * PESIZE + idy;
    	}
}

__global__ void sendToRandom(myCurandState_t *state,
							 int device_list[][PESIZE],
							 int temp_device_list[][PESIZE * FACTOR],
							 int random_list_counter[PENUMBER])
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;

	if (idx < PENUMBER && idy < PESIZE) {


		float r = getnextrandfloat(&state[idx * PESIZE + idy]);

		int random_position = r * (PENUMBER - 1);

		int acquired_position =
			atomicAdd(&random_list_counter[random_position], 1);

        if(acquired_position > PESIZE * FACTOR) {
          printf("BUG! PE %d has %d items already\n", random_position, random_list_counter[random_position]);
        }

		temp_device_list[random_position][acquired_position] =
			device_list[idx][idy];
	}
}


__global__ void clearCounter(int random_list_counter[PENUMBER])
{

	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx < PENUMBER) {
		random_list_counter[idx] = 0;

	}
}

__global__ void checkCounter(int random_list_counter[PENUMBER])
{

	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx < PENUMBER) {
		printf("counter at %d is %d\n", idx, random_list_counter[idx]);

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

	if (left == right)
		return;

	if (left > right) {
		right = 1 + right;
	}

	int stack_size = 0;
	sort_data stack[PESIZE * FACTOR];

	stack[stack_size++] = {
	left, right};


	while (stack_size > 0) {

		int curr_left = stack[stack_size - 1].left;
		int curr_right = stack[stack_size - 1].right;
		stack_size--;

		if (curr_left < curr_right) {
			int part = partition(data, curr_left, curr_right);
			stack[stack_size++] = {
			curr_left, part - 1};
			stack[stack_size++] = {
			part + 1, curr_right};
		}
	}
}



__global__ void sortList(int temp_device_list[][PESIZE * FACTOR],
						 int random_list_counter[PENUMBER])
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < PENUMBER) {


		int number = random_list_counter[idx];
        	if(number > PESIZE * FACTOR) {
          		printf("BUG! PE %d claims to have %d items\n", idx, number);
        	}

		if (number > 1) {

			quicksort_seq(temp_device_list[idx], number - 1);
		}

	}

}

__global__ void randomPermute(myCurandState_t *state,
							  int temp_device_list[][PESIZE * FACTOR],
							  int random_list_counter[PENUMBER])
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx < PENUMBER) {
		for (int i = 0; i < random_list_counter[idx]; i++) {


			float r = getnextrandfloat(&state[idx]);

			int j = r * (random_list_counter[idx] - 1);


			int temp = temp_device_list[idx][i];
			temp_device_list[idx][i] = temp_device_list[idx][j];
			temp_device_list[idx][j] = temp;
		}
	}

}



__global__ void recoverSize(int device_list[][PESIZE],
							int temp_device_list[][PESIZE * FACTOR],
							int random_list_counter[PENUMBER],
							int scanned_random_list_counter[PENUMBER],
                            int device_list_inverse[][SIZE])
{

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < PENUMBER) {

		int delta = scanned_random_list_counter[idx];

		for (int i = 0; i < random_list_counter[idx]; i++) {
			int addValue = delta + i;
			int interResult =
				PENUMBER * addValue / (PESIZE * PENUMBER);

	                int pe = interResult;
            		int index = (delta - PESIZE * interResult + i);

            		int index_linear = pe * PESIZE + index;

            		int val = temp_device_list[idx][i];

			device_list[pe][index] = val;

            		device_list_inverse[val / SIZE][val % SIZE] = index_linear;
		}
	}

}


int main(int argc, char *argv[])
{

    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1024 * 1024 * 1000);
    struct timespec start, stop;
    double accum;

    int (*device_permutation_list)[PESIZE];
    int (*device_permutation_list_inverse)[SIZE];
    int (*device_temp_permutation_list)[PESIZE * FACTOR];
    int *random_list_counter;
    int (*scanned_random_list_counter);
    
    int (*device_grid)[SIZE + 2];
    int (*device_newGrid)[SIZE + 2];
    int (*device_moveGrid)[SIZE + 2];

    int (*device_tempGrid)[SIZE + 2];

    int (*device_rowAndColumn)[SIZE + 2];

    int (*device_local_trials)[SIZE + 2];


    int seed = atoi(argv[1]);
	srand(seed);

	myCurandState_t *devState;

	cudaMalloc((void **) &device_permutation_list, sizeof(int) * (TOTAL));
#ifdef DEBUG
	cudaDeviceSynchronize();
	cudaCheckError();
#endif

	cudaMalloc((void **) &device_permutation_list_inverse, sizeof(int) * (TOTAL));
#ifdef DEBUG
	cudaDeviceSynchronize();
	cudaCheckError();
#endif

	cudaMalloc((void **) &device_temp_permutation_list, sizeof(int) * (TOTAL) * FACTOR);
#ifdef DEBUG
	cudaDeviceSynchronize();
	cudaCheckError();
#endif

    cudaMalloc(&random_list_counter, sizeof(int)*(PENUMBER));
#ifdef DEBUG
	cudaDeviceSynchronize();
	cudaCheckError();
#endif

    cudaMalloc((void**)&devState, GRIDTOTAL * sizeof(myCurandState_t));
#ifdef DEBUG
	cudaDeviceSynchronize();
	cudaCheckError();
#endif

    cudaMalloc((void**)&scanned_random_list_counter, sizeof(int) * PENUMBER);
#ifdef DEBUG
	cudaDeviceSynchronize();
	cudaCheckError();
#endif


    size_t bytes = sizeof(int) * (SIZE + 2) * (SIZE + 2);

    cudaMalloc((void **) &device_grid, bytes);
#ifdef DEBUG
	cudaDeviceSynchronize();
	cudaCheckError();
#endif

    cudaMalloc((void **) &device_newGrid, bytes);
#ifdef DEBUG
	cudaDeviceSynchronize();
	cudaCheckError();
#endif

    cudaMalloc((void **) &device_tempGrid, bytes);
#ifdef DEBUG
	cudaDeviceSynchronize();
	cudaCheckError();
#endif

    cudaMalloc((void **) &device_moveGrid, bytes);
#ifdef DEBUG
	cudaDeviceSynchronize();
	cudaCheckError();
#endif

    cudaMalloc((void **) &device_rowAndColumn, bytes);
#ifdef DEBUG
	cudaDeviceSynchronize();
	cudaCheckError();
#endif

    cudaMalloc((void **) &device_local_trials, bytes);
#ifdef DEBUG
	cudaDeviceSynchronize();
	cudaCheckError();
#endif

	int blockSizeVerPermu = numThreadsPerBlock / PESIZE;
	dim3 blockSizePermu(blockSizeVerPermu, PESIZE, 1);


	initCurand <<< GRIDTOTAL / 1024,
		1024 >>> (devState, seed * TOTAL);

#ifdef DEBUG
	cudaDeviceSynchronize();
	cudaCheckError();
#endif


	initPermutationList <<< ceil((float)(PENUMBER * PESIZE) / numThreadsPerBlock), blockSizePermu >>> (device_permutation_list);


#ifdef DEBUG
	cudaDeviceSynchronize();
	cudaCheckError();
#endif

	int blockSizePerDim = sqrt(numThreadsPerBlock);
	int gridSizePerDim = (SIZE + 2) / blockSizePerDim;

	dim3 blockSize(blockSizePerDim, blockSizePerDim, 1);
	dim3 gridSize(gridSizePerDim, gridSizePerDim, 1);


#ifdef DEBUG
	cudaDeviceSynchronize();
	cudaCheckError();
#endif

	if (clock_gettime(CLOCK_REALTIME, &start) == -1) {
		perror("clock gettime");
		exit(EXIT_FAILURE);
	}


    for (int i = 0; i < (SIZE + 2); i++) {
        for (int j = 0; j < SIZE + 2; j++) {
            host_grid[i][j] = 0;
        }
    }

    initPos(host_grid);

    cudaMemcpy(device_grid, host_grid, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_newGrid, host_grid, bytes, cudaMemcpyHostToDevice);



    newTogrid << <gridSize, blockSize >> >(device_grid, device_newGrid);

    update << <gridSize, blockSize >> >(device_tempGrid, device_moveGrid,
                                        device_rowAndColumn);

    int oneDimGridSize = ceil(PENUMBER / double(numThreadsPerBlock));

	cached_allocator alloc;
	for (int i = 0; i < ITERATIONS; i++) {


        	clearCounter<<<oneDimGridSize, (numThreadsPerBlock)>>>(random_list_counter);

#ifdef DEBUG
		cudaDeviceSynchronize();
		cudaCheckError();
#endif

		sendToRandom <<< ceil((TOTAL) / double (numThreadsPerBlock)),
			blockSizePermu >>> (devState, device_permutation_list,
								device_temp_permutation_list,
								random_list_counter);

#ifdef DEBUG
		cudaDeviceSynchronize();
		cudaCheckError();
#endif

        	int OneDimGridSize = ceil(PENUMBER / double(numThreadsPerBlock));
		sortList <<< OneDimGridSize,
			(numThreadsPerBlock) >>> (device_temp_permutation_list,
									  random_list_counter);
#ifdef DEBUG
		cudaDeviceSynchronize();
		cudaCheckError();
#endif
		randomPermute <<< OneDimGridSize,
			(numThreadsPerBlock) >>> (devState,
									  device_temp_permutation_list,
									  random_list_counter);


#ifdef DEBUG
		cudaDeviceSynchronize();
		cudaCheckError();
#endif


		thrust::exclusive_scan(thrust::cuda::par(alloc),
							   random_list_counter,
							   random_list_counter + PENUMBER,
							   scanned_random_list_counter);

#ifdef DEBUG
		cudaDeviceSynchronize();
		cudaCheckError();
#endif
		recoverSize <<< ceil(((double)PENUMBER) / numThreadsPerBlock),
			numThreadsPerBlock >>> (device_permutation_list,
									  device_temp_permutation_list,
									  random_list_counter,
									  scanned_random_list_counter,
                                      device_permutation_list_inverse);


#ifdef DEBUG
		cudaDeviceSynchronize();
		cudaCheckError();
#endif


        // all of the above was just to get the permuted priorities
        // now, the actual model code is called

        compute << <gridSize, blockSize >> >(device_grid, device_newGrid,
                                             device_tempGrid,
                                             device_moveGrid,
                                             device_permutation_list);

#ifdef DEBUG
        cudaDeviceSynchronize();
        cudaCheckError();
#endif

        prepareNewGrid <<< gridSize, blockSize >>> (device_tempGrid,
                                                    device_newGrid,
                                                    device_local_trials);

        assign << <gridSize, blockSize >> >(devState, device_grid,
                                            device_newGrid,
                                            device_tempGrid,
                                            device_moveGrid,
                                            device_rowAndColumn,
                                            device_permutation_list,
                                            device_permutation_list_inverse,
                                            device_local_trials);
#ifdef DEBUG
        cudaDeviceSynchronize();
        cudaCheckError();
#endif

        updateTonew << <gridSize, blockSize >> >(device_grid,
                                                 device_newGrid,
                                                 device_tempGrid,
                                                 device_moveGrid,
                                                 device_rowAndColumn,
                                                 device_permutation_list_inverse);
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

	printf("%.1f Time is %.5f s\n", float(OCCUPANCY),accum / 1e6);

    cudaMemcpy(host_grid, device_newGrid, bytes, cudaMemcpyDeviceToHost);
    //printOutput(host_grid);
    //checkOutput(host_grid);

	return 0;


}



