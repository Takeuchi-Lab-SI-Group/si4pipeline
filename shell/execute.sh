#!/bin/bash

#PBS -V
#PBS -S /bin/bash
#PBS -j oe
#PBS -o path_for_logs
#PBS -N pipeline
#PBS -q cpu-f01
#PBS -l select=1:ncpus=32:mem=128gb:ompthreads=1

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd $PBS_O_WORKDIR
uv run experiment/$MYFILE $MYARGS
