set title "Learning Curve"
set xlabel "Episode"
set ylabel "Return"
set term png
set output "learning_curve_zeroth_order.png"

plot "zeroth_order_log.txt" using 2:3 with lines title "Zeroth-Order Method"
