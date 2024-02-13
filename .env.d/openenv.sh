#!/bin/bash

#source .venv/bin/activate
conda activate adagdvb

# AdaGDVB
if [ -z ${AdaGDVB} ]; then
  export AdaGDVB=`pwd`
fi

export ROOT=`pwd`

# Libraries
export GDVB="${AdaGDVB}/lib/GDVB"
export R4V="${GDVB}/lib/R4V"
export DNNV="${GDVB}/lib/DNNV"
export DNNV_wb="${GDVB}/lib/DNNV_wb"
export DNNF="${GDVB}/lib/DNNF"
export SwarmHost="${GDVB}/lib/SwarmHost"

# Path
export PYTHONPATH="${PYTHONPATH}:${AdaGDVB}"
export PYTHONPATH="${PYTHONPATH}:${GDVB}"
export PYTHONPATH="${PYTHONPATH}:${R4V}"
export PYTHONPATH="${PYTHONPATH}:${SwarmHost}"

# misc
export acts_path="${GDVB}/lib/acts.jar"

alias adagdvb="python -m adagdvb"

export CONDA_HOME=$HOME/Apps/MiniConda3
export MKL_SERVICE_FORCE_INTEL=1

alias rmca='rm -rf results/*/ca*'
alias rmfg='rm -rf results/*/figures*'
