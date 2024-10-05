#!/bin/bash

file="summary.py"

qsub -v \
MYARGS="",MYFILE=$file \
shell/execute.sh
