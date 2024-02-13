#!/usr/bin/env python

import os
import sys

sorc = "results/MNIST.0/"
dest = "/home/edward/Dropbox/Apps/Overleaf/EvoGDVB_cav24/img_final/"

assert os.path.exists(sorc), sorc
assert os.path.exists(dest), dest


fig_dirs = [x for x in os.listdir(sorc) if "figures_" in x]

for fd in fig_dirs:
    v = fd.split("_")[1]

    name = "all_EvoState.Refine_1_Direction.Maintain.pdf"
    source = os.path.join(sorc, fd, name)
    target = os.path.join(dest, f"{v}.pdf")

    cmd = f"cp {source} {target}"
    print(cmd)
    if len(sys.argv) >=2 and sys.argv[1] == 'go':
        os.system(cmd)
