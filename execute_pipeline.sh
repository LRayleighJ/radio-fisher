#!/bin/bash

a=(22 26 29)
for i in ${a[*]}       
do 
   python /home/zerui603/work/bao21cm-master/full_experiment.py $i
done

#python /home/zerui603/work/bao21cm-master/plotting/output_fisher_matrix.py
#python /home/zerui603/work/bao21cm-master/plotting/plot_dlogp.py
#python /home/zerui603/work/bao21cm-master/plotting/plot_w0wa.py