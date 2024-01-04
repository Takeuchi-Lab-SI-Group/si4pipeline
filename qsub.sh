#!/bin/csh
#$ -cwd
#$ -V -S /bin/bash
#$ -N cpu_e
#$ -q cpu-e.q@*
#$ -pe smp 32
#$ -o path_for_stdoutfile
#$ -e path_for_errorfile
# export OMP_NUM_THREADS = 1

$@
