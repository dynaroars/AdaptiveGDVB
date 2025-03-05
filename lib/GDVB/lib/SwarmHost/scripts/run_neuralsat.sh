#!/bin/bash

. ${SwarmHost}/scripts/init_conda.sh
#conda activate neuralsat
conda activate neuralsat2

which python
export MKL_SERVICE_FORCE_INTEL=1

#cmd="python $SwarmHost/lib/neuralsat/src/main.py $@"
#cmd="python $SwarmHost/lib/neuralsat/neuralsat/main.py $@"
cmd="python $SwarmHost/lib/neuralsat/neuralsat-pt201/main.py $@"
echo $cmd
$cmd

conda deactivate
