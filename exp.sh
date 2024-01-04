# bash

for i in {0..9}; do
    qsub qsub.sh python test.py \
        --seed $i \
        >> result.txt 2>&1
done
