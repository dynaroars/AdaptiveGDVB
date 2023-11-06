#!/bin/bash

#source .venv/bin/activate
conda activate evogdvb

# EvoGDVB
export EvoGDVB=`pwd`

# Libraries
export GDVB="${EvoGDVB}/lib/GDVB"
export R4V="${EvoGDVB}/lib/R4V"
export DNNV="${EvoGDVB}/lib/DNNV"
export DNNV_wb="${EvoGDVB}/lib/DNNV_wb"
export DNNF="${EvoGDVB}/lib/DNNF"
export SwarmHost="${EvoGDVB}/lib/SwarmHost"

# Path
export PYTHONPATH="${PYTHONPATH}:${EvoGDVB}"
export PYTHONPATH="${PYTHONPATH}:${GDVB}"
export PYTHONPATH="${PYTHONPATH}:${R4V}"
export PYTHONPATH="${PYTHONPATH}:${SwarmHost}"

# misc
export acts_path="${GDVB}/lib/acts.jar"
export GRB_LICENSE_FILE="${GDVB}/lib/gurobi.lic"

alias evogdvb="python -m evogdvb"

export CONDA_HOME=$HOME/Apps/MiniConda3
