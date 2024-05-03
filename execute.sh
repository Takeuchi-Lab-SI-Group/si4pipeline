#!/bin/bash

#PBS -V
#PBS -S /bin/bash
#PBS -j oe
#PBS -o path_for_logs
#PBS -N pl
#PBS -q cpu4-6
#PBS -l select=1:ncpus=32:mem=128gb:ompthreads=1

cd $PBS_O_WORKDIR
python experiment/$MYFILE $MYARGS
