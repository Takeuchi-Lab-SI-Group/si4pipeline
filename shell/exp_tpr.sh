#!/bin/bash

file="experiment_main.py"

for option in op1 op2 all_cv; do
    for delta in 0.2 0.4 0.6 0.8; do
        qsub \
        -v \
        MYARGS="--delta $delta --option $option",MYFILE=$file \
        shell/execute.sh
    done
done
