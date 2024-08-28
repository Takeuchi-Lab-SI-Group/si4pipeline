file="experiment_main.py"

qsub \
-v MYARGS="--seed 0 --n 100 --d 20 --option op1",MYFILE=$file \
execute.sh
