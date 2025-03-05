#!/bin/bash

. ${SwarmHost}/scripts/init_conda.sh
conda activate mnbab

MNBAB=$SwarmHost/lib/mnbab
cd $MNBAB
export PYTHONPATH=$PYTHONPATH:$PWD
echo $PYTHONPATH

cmd="python $SwarmHost/swarm_host/verifiers/mnbab/exe.py $@"
echo $cmd
$cmd

cd $OCTOPUS
conda deactivate