set terminal postscript eps enhanced size 3.0in,1.2in font 'Helvetica,10'
#set terminal postscript eps size 4in,2in font 'Helvetica,10'
set output 'out.eps'

#set log x 2
#set log y

set key bottom right

set xlabel "Occupancy ratio"
set ylabel "Execution time [s]"

#set ylabel "Rel. Reduction in Updates"
#set y2label "Speedup"

set xtics nomirror
#set ytics nomirror
#set y2tics nomirror

#set grid y2
set grid x
set grid y

#set xrange [0.5:256]
#set yrange [0:100]
#set yrange [0:6]
set yrange [0:16]
#set yrange [0:2.0]

set datafile separator ","

plot 'output_combined.txt' using ($1):6:7 with yerrorlines lt 4 lw 1 pt 12 title "Sampling and permutation",\
'' using ($1):2:3 with yerrorlines lt 1 lw 1 pt 6 title "Iterative push incremental tie-braking",\
''using ($1):8:9  with linespoints lt 4 lw 1 pt 8 title "Iterative push postponed tie-braking",\
'' using ($1):4:5 with yerrorlines lt 5 lw 1 pt 3 title "Non-iterative push incremental tie-braking",\

#''plot  'out_ci_6.dat' using 1:2:3  with yerrorlines lt 1 lw 2 title "Time deter_6"
#'' using 1:6:7 with xerrorbars lt 1 lw 1 title "SP"

