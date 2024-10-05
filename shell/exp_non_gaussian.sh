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
