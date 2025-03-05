#!/bin/bash

conda activate swarmhost

if [ -z ${SwarmHost} ]; then
  export SwarmHost=`pwd`
fi

export ROOT=`pwd`

alias swarmhost="python -m swarm_host"
