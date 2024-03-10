# bash

machin_min=6
machin_max=10

machin_index=0
divisor=$(($machin_max-$machin_min+1))

file="base_experiment.py"

for option in op1 op2; do
    for delta in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8; do
        machin=$(printf "%02d" $((($machin_index%$divisor)+$machin_min)))
        qsub \
        -v MYARGS="--num_results 1000 --num_worker 96 --delta $delta --option $option",MYFILE=$file \
        shell/f$machin.sh
        machin_index=$(($machin_index+1))
    done
done
