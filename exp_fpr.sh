# bash

file="main_experiment.py"

for option in op1 op2 op1cv op2cv op12cv; do
    for seed in $(seq 0 3); do
        for n in 400 300 200 100; do
            qsub \
            -v MYARGS="--seed $seed --n $n --option $option",MYFILE=$file \
            execute.sh
        done
        for p in 40 30 10; do
            qsub \
            -v MYARGS="--seed $seed --p $p --option $option",MYFILE=$file \
            execute.sh
        done
    done
done
