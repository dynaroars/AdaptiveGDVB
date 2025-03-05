#!/bin/bash


. ${SwarmHost}/scripts/init_conda.sh
conda activate veristable

which python
export MKL_SERVICE_FORCE_INTEL=1

cmd="python $SwarmHost/lib/veristable/src/main.py $@"
echo $cmd
$cmd

conda deactivate
