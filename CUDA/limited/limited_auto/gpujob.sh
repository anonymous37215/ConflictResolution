#!/bin/sh

echo Performance > output_iterative_push_postponed.txt
for SIZE in 4094; do
	for agentNumber in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
	
                sed  -i -e "s/#define SIZE .*/#define SIZE $SIZE/" ../iterative_push_postponed/parameter.cuh
                sed  -i -e "s/const int agentNumber .*/const int agentNumber = ((int)($agentNumber * SIZE * SIZE));/" ../iterative_push_postponed/parameter.cuh
                sed  -i -e "s/#define OCCUPANCY .*/#define OCCUPANCY $agentNumber/" ../iterative_push_postponed/parameter.cuh

                nvcc  -std=c++11 ../iterative_push_postponed/iterative_push_postponed.cu
                for i in 1 2 3 4 5 6 7 8 9 10; do
		#for i in 1; do
                            ./a.out 100 >> output_iterative_push_postponed.txt
		done
        done
done

echo Performance > output_noniterative_push_incremental.txt
for SIZE in 4094; do
        for agentNumber in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
                sed  -i -e "s/#define SIZE .*/#define SIZE $SIZE/" ../noniterative_push_incremental/parameter.cuh
                sed  -i -e "s/const int agentNumber .*/const int agentNumber = ((int)($agentNumber * SIZE * SIZE));/" ../noniterative_push_incremental/parameter.cuh
                sed  -i -e "s/#define OCCUPANCY .*/#define OCCUPANCY $agentNumber/" ../noniterative_push_incremental/parameter.cuh

                nvcc -std=c++11 ../noniterative_push_incremental/noniterative_push_incremental.cu
                for i in 1 2 3 4 5 6 7 8 9 10; do
                #for i in 1; do
                            ./a.out 100 >> output_noniterative_push_incremental.txt
                done
        done
done


echo Performance > output_iterative_push_incremental.txt
for SIZE in 4094; do
        for agentNumber in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
        	sed  -i -e "s/#define SIZE .*/#define SIZE $SIZE/" ../iterative_push_incremental/parameter.cuh
                sed  -i -e "s/const int agentNumber .*/const int agentNumber = ((int)($agentNumber * SIZE * SIZE));/" ../iterative_push_incremental/parameter.cuh
                sed  -i -e "s/#define OCCUPANCY .*/#define OCCUPANCY $agentNumber/" ../iterative_push_incremental/parameter.cuh

                nvcc -std=c++11 ../iterative_push_incremental/iterative_push_incremental.cu
                for i in 1 2 3 4 5 6 7 8 9 10; do
                #for i in 1; do
                            ./a.out 1 >> output_iterative_push_incremental.txt
                done
        done
done



echo Performance > output_iterative_pull_postponed.txt
for SIZE in 4094; do
        for agentNumber in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
        	sed  -i -e "s/#define SIZE .*/#define SIZE $SIZE/" ../iterative_pull_postponed/parameter.cuh
                sed  -i -e "s/const int agentNumber .*/const int agentNumber = ((int)($agentNumber * SIZE * SIZE));/" ../iterative_pull_postponed/parameter.cuh
                sed  -i -e "s/#define OCCUPANCY .*/#define OCCUPANCY $agentNumber/" ../iterative_pull_postponed/parameter.cuh

                nvcc -std=c++11 ../iterative_pull_postponed/iterative_pull_postponed.cu
                for i in 1 2 3 4 5 6 7 8 9 10; do
                #for i in 1; do
                           ./a.out 100 >> output_iterative_pull_postponed.txt
                done
        done
done

