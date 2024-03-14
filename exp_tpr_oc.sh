# bash

file="main_experiment.py"

for option in op1 op2 op1cv op2cv op12cv; do
    for seed in $(seq 0 3); do
        for delta in 0.2 0.4 0.6 0.8; do
            qsub \
            -v MYARGS="--seed $seed --delta $delta --option $option --oc oc",MYFILE=$file \
            execute.sh
        done
    done
done
