#!/bin/bash

echo $@
args=("$@")
bash -c "${args[$SLURM_PROCID]}"
#zsh -i -c "${args[$SLURM_PROCID]}"
