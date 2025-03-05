#!/bin/bash


. ${SwarmHost}/scripts/init_conda.sh
conda activate abcrown23

which python
export MKL_SERVICE_FORCE_INTEL=1

cmd="python $SwarmHost/lib/abcrown23/complete_verifier/abcrown.py $@"
echo $cmd
$cmd

conda deactivate
