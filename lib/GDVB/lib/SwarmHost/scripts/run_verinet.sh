#!/bin/bash

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/sw/ubuntu-22.04/anaconda3/2023.03/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/sw/ubuntu-22.04/anaconda3/2023.03/etc/profile.d/conda.sh" ]; then
        . "/sw/ubuntu-22.04/anaconda3/2023.03/etc/profile.d/conda.sh"
    else
        export PATH="/sw/ubuntu-22.04/anaconda3/2023.03/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
conda deactivate

conda activate verinet

Verinet=$SwarmHost/lib/verinet
cd $Verinet
export PYTHONPATH=$PYTHONPATH:$PWD
echo $PYTHONPATH
cd $OCTOPUS

cmd="python $SwarmHost/swarm_host/verifiers/verinet/exe.py $@"
echo $cmd
$cmd

conda deactivate