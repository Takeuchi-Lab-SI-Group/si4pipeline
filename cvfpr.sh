# bash

machin_min=8
machin_max=10

machin_index=0
divisor=$(($machin_max-$machin_min+1))

file="cv_experiment.py"


for option in op1and2; do
    for n in 400 300 200 100; do
        machin=$(printf "%02d" $((($machin_index%$divisor)+$machin_min)))
        qsub \
        -v MYARGS="--num_results 1000 --num_worker 96 --n $n --cv_mode $option",MYFILE=$file \
        shell/f$machin.sh
        machin_index=$(($machin_index+1))
    done
done

for option in op1and2; do
    for p in 40 30 10; do
        machin=$(printf "%02d" $((($machin_index%$divisor)+$machin_min)))
        qsub \
        -v MYARGS="--num_results 1000 --num_worker 96 --p $p --cv_mode $option",MYFILE=$file \
        shell/f$machin.sh
        machin_index=$(($machin_index+1))
    done
done
