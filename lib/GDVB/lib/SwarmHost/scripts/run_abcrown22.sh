#!/bin/bash


. ${SwarmHost}/scripts/init_conda.sh
conda activate abcrown

which python
export MKL_SERVICE_FORCE_INTEL=1

cmd="python $SwarmHost/lib/abcrown/complete_verifier/abcrown.py $@"
echo $cmd
$cmd

conda deactivate
