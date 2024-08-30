#!/bin/bash

file="experiment_time.py"

ns=(400 800 800 800 800 1200 1600)
ds=(80  40  80  120 160   80   80)

for option in op1 op1_parallel op1_serial; do
    for i in ${!ns[@]}; do
        n=${ns[i]}
        d=${ds[i]}

        qsub \
        -v \
        MYARGS="--n $n --d $d --option $option",MYFILE=$file \
        shell/execute.sh
    done
done
