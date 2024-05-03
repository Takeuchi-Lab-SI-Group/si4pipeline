# bash

file="../robust/robust_experiment_nongauss.py"

for option in op1 op2; do
    for seed in $(seq 2 9); do
        for noise in skewnorm exponnorm gennormsteep gennormflat t; do
            for distance in 0.01 0.02 0.03 0.04; do
                qsub \
                -v MYARGS="--seed $seed --noise $noise --distance $distance --option $option",MYFILE=$file \
                execute.sh
            done
        done
    done
done
