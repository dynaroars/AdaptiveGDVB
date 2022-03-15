#!/usr/bin/env bash

python -m evogdvb configs_evo/edz.toml evolutionary --seed 10 --result_dir res_s3 2>./tmp/edz.err
# python -m evogdvb configs_evo/edp.toml evolutionary --seed 10 --result_dir res_s3 2>./tmp/edp.err
# python -m evogdvb configs_evo/neurify.toml evolutionary --seed 10 --result_dir res_s3 2>./tmp/neurify.err
# python -m evogdvb configs_evo/verinet.toml evolutionary --seed 10 --result_dir res_s3 2>./tmp/verinet.err
# python -m evogdvb configs_evo/dnnf.toml evolutionary --seed 10 --result_dir res_s3 2>./tmp/dnnf.err
# python -m evogdvb configs_evo/nnenum.toml evolutionary --seed 10 --result_dir res_s3 2>./tmp/nnenum.err
