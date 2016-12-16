set output "pi_scale_out.eps"
set terminal postscript eps enhanced monochrome font 'Helvetica,20'

set style line 1 lt 1 lw 2 pt 3 ps 2
set style line 2 lt 1 lw 2 pt 4 ps 2
set style line 3 lt 1 lw 2 pt 7 ps 2
set style line 4 lt 1 lw 2 pt 6 ps 2
set style line 4 lt 1 lw 2 pt 5 ps 2
set key top left
set xlabel "number of particles"
set ylabel "Total running time (second)"
set xtics autofreq nomirror
set xrange [0:5000]
plot 'clustersize2.dat' using 1:2:xtic(1) with linespoints ls 1 t "single-threaded", \
'clustersize2.dat' using 1:3:xtic(1) with linespoints ls 2 t "OpenMP 1 threads", \
'clustersize2.dat' using 1:4:xtic(1) with linespoints ls 3 t "OpenMP 2 threads", \
'clustersize2.dat' using 1:5:xtic(1) with linespoints ls 4 t "OpenMP 4 threads", \
'clustersize2.dat' using 1:6:xtic(1) with linespoints ls 5 t "OpenMP 8 threads", \
