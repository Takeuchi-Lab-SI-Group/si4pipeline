#!/bin/bash

file="experiment_real.py"


for key in heating_load cooling_load gas_turbine red_wine white_wine abalone concrete housing; do
    for n in 100 150 200; do
        qsub \
        -v \
        MYARGS="--key $key --n $n",MYFILE=$file \
        shell/execute.sh
    done
done
