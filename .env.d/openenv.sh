#!/bin/bash

source .venv/bin/activate

# EvoGDVB
export EvoGDVB=`pwd`
if [ -z ${EvoGDVB} ]; then
  export EvoGDVB=`pwd`
fi
# export TMPDIR=$EvoGDVB/tmp

# Libraries
export GDVB="${EvoGDVB}/lib/GDVB"
export R4V="${GDVB}/lib/R4V"
export DNNV="${GDVB}/lib/DNNV"
export DNNV_wb="${GDVB}/lib/DNNV_wb"
export DNNF="${GDVB}/lib/DNNF"

# Path
export PYTHONPATH="${PYTHONPATH}:${EvoGDVB}"
export PYTHONPATH="${PYTHONPATH}:${GDVB}"
export PYTHONPATH="${PYTHONPATH}:${GDVB}/lib/R4V/"

# misc
export acts_path="${GDVB}/lib/acts.jar"
export GRB_LICENSE_FILE="${GDVB}/lib/gurobi.lic"

alias evogdvb="python -m evogdvb"
