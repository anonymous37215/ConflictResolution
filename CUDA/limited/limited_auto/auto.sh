#!/bin/sh

./get_ci.pl output_iterative_push_incremental.txt > output_iterative_push_incremental_ci.txt
./get_ci.pl output_noniterative_push_incremental.txt > output_noniterative_push_incremental_ci.txt
./get_ci.pl output_iterative_push_postponed.txt > output_iterative_push_postponed_ci.txt
./get_ci.pl output_iterative_pull_postponed.txt > output_iterative_pull_postponed_ci.txt

python format.py
gnuplot gnuplot_example_limited.plot

