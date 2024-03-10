#!/bin/bash

#PBS -V
#PBS -S /bin/bash
#PBS -j oe
#PBS -o path_for_logs
#PBS -N pl
#PBS -l select=1:ncpus=96:mem=256gb:host=cpu-f07:ompthreads=1

cd $PBS_O_WORKDIR
python experiment/$MYFILE $MYARGS
