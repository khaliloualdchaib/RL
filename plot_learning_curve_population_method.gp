set title "Learning Curve"
set xlabel "Episode"
set ylabel "Return"
set term png
set output "learning_curve_population_method.png"

plot "population_method_log.txt" using 2:3 with lines title "Zeroth-Order Method"
