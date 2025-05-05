#!/bin/bash

. ${SwarmHost}/scripts/init_conda.sh

conda run -n marabou python $SwarmHost/lib/Marabou/resources/runMarabou.py $@
