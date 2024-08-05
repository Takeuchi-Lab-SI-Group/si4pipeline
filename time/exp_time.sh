# bash

file="../time/time_experiment.py"

for seed in $(seq 1 3); do
    for option in default parallel serial; do
        for n in 400 800 1200 1600; do
            qsub \
            -v MYARGS="--seed $seed --n $n --p 80 --option $option",MYFILE=$file \
            execute.sh
        done
        for p in 40 120 160; do
            qsub \
            -v MYARGS="--seed $seed --n 800 --p $p --option $option",MYFILE=$file \
            execute.sh
        done
    done
done
