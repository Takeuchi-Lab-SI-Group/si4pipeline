# bash

machin_min=6
machin_max=10

machin_index=0
divisor=$(($machin_max-$machin_min+1))

file="base_experiment.py"

for option in op1 op2; do
    for n in 400 300 200 100; do
        machin=$(printf "%02d" $((($machin_index%$divisor)+$machin_min)))
        qsub \
        -v MYARGS="--num_results 2000 --num_worker 96 --n $n --option $option",MYFILE=$file \
        shell/f$machin.sh
        machin_index=$(($machin_index+1))
    done
done

for option in op1 op2; do
    for p in 40 30 10; do
        machin=$(printf "%02d" $((($machin_index%$divisor)+$machin_min)))
        qsub \
        -v MYARGS="--num_results 2000 --num_worker 96 --p $p --option $option",MYFILE=$file \
        shell/f$machin.sh
        machin_index=$(($machin_index+1))
    done
done
