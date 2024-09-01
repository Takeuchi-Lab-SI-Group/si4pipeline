#!/bin/bash

file="experiment_time.py"

ns=(1600 1200 800 800 800 800 400)
ds=(  80   80 160 120  80  40  80)

for option in op1_serial op1_parallel op1; do
    for i in ${!ns[@]}; do
        n=${ns[i]}
        d=${ds[i]}

        qsub \
        -v \
        MYARGS="--n $n --d $d --option $option",MYFILE=$file \
        shell/execute.sh
    done
done
