#!/bin/bash

#SBATCH --nodes 1

#SBATCH --output slurm_outputs/slurm-%j.out
srun hostname
a=''
while (( "$#" )); do
  a="$a '$1'"
  shift
done
srun bash -c "./run_script/run_general_supp.sh $a"

