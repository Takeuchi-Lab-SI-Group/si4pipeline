#!/bin/bash

username="shiraishi.t"
job_lines=$(qstat | awk '$3 == "'"$username"'" {print}')
job_ids=$(echo "$job_lines" | awk '{print $1}' | tr '\n' ' ')

for job_id in $job_ids; do  # forループで各ジョブIDに対してqdelを実行
    qdel "$job_id"
done
