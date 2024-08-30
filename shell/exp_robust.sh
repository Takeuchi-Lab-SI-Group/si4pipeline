#!/bin/bash

file="experiment_robust.py"

for seed in {0..9}; do
    for distance in 0.01 0.02 0.03 0.04; do
        for robust_type in exponnorm skewnorm gennormsteep gennormflat t; do
            qsub \
            -v \
            MYARGS="--seed $seed --robust_type $robust_type --distance $distance",MYFILE=$file \
            shell/execute.sh
        done
    done
done

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
