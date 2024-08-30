#!/bin/bash

file="experiment_robust.py"

ns=(400 300 200 200 200 200 100)
ds=(20  20  40  30  20  10  20)

for seed in {0..9}; do
    for i in ${!ns[@]}; do
        n=${ns[i]}
        d=${ds[i]}

        qsub \
        -v \
        MYARGS="--seed $seed --n $n --d $d",MYFILE=$file \
        shell/execute.sh
    done
done
