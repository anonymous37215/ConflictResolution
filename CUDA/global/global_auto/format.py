#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt



with open('output_iterative_push_ci.txt') as f1:
	lines1 = f1.readlines()



with open('output_noniterative_push_ci.txt') as f2:
        lines2 = f2.readlines()


with open('output_sampling_permutation_ci.txt') as f3:
        lines3 = f3.readlines()

with open('output_iterative_push_postponed_ci.txt') as f4:
        lines4 = f4.readlines()





for n,line in enumerate(lines1):
	#if n % 2 == 1:
	#if line.startswith("Time"):
	lines1[n] = line.rstrip() + ',' + lines2[n].split(',')[1] + ',' + lines2[n].split(',')[2].rstrip() + ',' + lines3[n].split(',')[1] + ',' + lines3[n].split(',')[2].rstrip() + ',' + lines4[n].split(',')[1] + ',' + lines4[n].split(',')[2]
	#else:
		#lines[n]="\n" + line.rstrip()
#print ' '.join(lines)


with open('output_combined.txt', 'w') as wfile:
    wfile.write(' '.join(lines1))
