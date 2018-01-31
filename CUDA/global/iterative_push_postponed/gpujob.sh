#!/bin/sh

#SBATCH -o gpu-job-%j.output
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -n 1

module load cuda75
module load gcc/4.8.5

echo Performance > output.txt
for SIZE in 4094; do
	for agentNumber in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
#        for PESIZE in 2 4 8 16 32 64; do
#        for numThreadsPerBlock in 64 256; do
#                for SAM_PESIZE in 1024 2048 4096; do
#                        for SAM_numThreadsPerBlock in 256 512 1024; do
                            sed  -i -e "s/#define SIZE .*/#define SIZE $SIZE/" parameter.cuh
                            sed  -i -e "s/const int agentNumber .*/const int agentNumber = ((int)($agentNumber * SIZE * SIZE));/" parameter.cuh
                            sed  -i -e "s/#define OCCUPANCY .*/#define OCCUPANCY $agentNumber/" parameter.cuh

			    #sed  -i -e "s/#define PESIZE .*/#define PESIZE $PESIZE/" parameter.cuh
                            #sed  -i -e "s/#define numThreadsPerBlock .*/#define numThreadsPerBlock $numThreadsPerBlock/" parameter.cuh
                            #sed  -i -e "s/#define SAM_PESIZE .*/#define SAM_PESIZE $SAM_PESIZE/" parameter.cuh
                            #sed  -i -e "s/#define SAM_numThreadsPerBlock .*/#define SAM_numThreadsPerBlock $SAM_numThreadsPerBlock/" parameter.cuh
                            #echo parameters: $SIZE $PESIZE $numThreadsPerBlock $SAM_PESIZE $SAM_numThreadsPerBlock
                            /cm/shared/apps/cuda75/toolkit/7.5.18/bin/nvcc --compiler-bindir /cm/shared/apps/gcc/4.8.5/bin -std=c++11 schelling_gpu.cu
			    #echo $i >> output.txt
               for i in 1 2 3 4 5 6 7 8 9 10; do
		#for i in 1; do
			    #echo $agentNumber >> output_3.txt	
                            ./a.out 100 >> output.txt
		done
        done
done
 #         done
 #       done
#done

