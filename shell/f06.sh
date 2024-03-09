#!/bin/bash

#PBS -V
#PBS -S /bin/bash
#PBS -j oe
#PBS -o path_for_logs
#PBS -N ptmc
#PBS -l select=1:ncpus=32:mem=64gb:host=cpu-f06:ompthreads=1

cd $PBS_O_WORKDIR
python experiment/$MYFILE $MYARGS
