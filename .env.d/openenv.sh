#!/bin/bash

#source .venv/bin/activate
conda activate evogdvb

# EvoGDVB
if [ -z ${EvoGDVB} ]; then
  export EvoGDVB=`pwd`
fi

export ROOT=`pwd`

# Libraries
export GDVB="${EvoGDVB}/lib/GDVB"
export R4V="${GDVB}/lib/R4V"
export DNNV="${GDVB}/lib/DNNV"
export DNNV_wb="${GDVB}/lib/DNNV_wb"
export DNNF="${GDVB}/lib/DNNF"
export SwarmHost="${GDVB}/lib/SwarmHost"

# Path
export PYTHONPATH="${PYTHONPATH}:${EvoGDVB}"
export PYTHONPATH="${PYTHONPATH}:${GDVB}"
export PYTHONPATH="${PYTHONPATH}:${R4V}"
export PYTHONPATH="${PYTHONPATH}:${SwarmHost}"

# misc
export acts_path="${GDVB}/lib/acts.jar"

alias evogdvb="python -m evogdvb"

export CONDA_HOME=$HOME/Apps/MiniConda3

export MKL_SERVICE_FORCE_INTEL=1

#export train_nodes_exclude="""cheetah01,ai01,sds01,sds02,lynx10,lynx12"""
#export verify_nodes="cortado03,cortado04,cortado05,cortado06,cortado07,cortado08,cortado09,cortado10"

alias rmca='rm -rf results/*/ca*'
alias rmfg='rm -rf results/*/figures*'
