# bash

file="../real/real_experiment.py"


for mode in heating_load cooling_load gas_turbin red_wine white_wine abalone concrete housing; do
    for option in op1 op2; do
        for n in 25; do
            for seed in $(seq 0 0); do
                qsub \
                -v MYARGS="--seed $seed --mode $mode --option $option --n $n",MYFILE=$file \
                execute.sh
            done
        done
    done
done
