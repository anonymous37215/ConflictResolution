#!/bin/sh

./get_ci.pl output_iterative_push.txt > output_iterative_push_ci.txt
./get_ci.pl output_noniterative_push.txt > output_noniterative_push_ci.txt
./get_ci.pl output_sampling_permutation.txt > output_sampling_permutation_ci.txt
./get_ci.pl output_iterative_push_postponed.txt > output_iterative_push_postponed_ci.txt
python format.py
gnuplot gnuplot_example.plot

