#!/usr/bin/env bash

# neu fc
./scripts/run_GDVB.sh configs/s2_mnist_fc.toml analyze --seed 10 --result_dir res_s1
./scripts/run_GDVB.sh configs/s2_mnist_fc.toml analyze --seed 11 --result_dir res_s1
./scripts/run_GDVB.sh configs/s2_mnist_fc.toml analyze --seed 12 --result_dir res_s1
./scripts/run_GDVB.sh configs/s2_mnist_fc.toml analyze --seed 13 --result_dir res_s1
./scripts/run_GDVB.sh configs/s2_mnist_fc.toml analyze --seed 14 --result_dir res_s1

# neu idm
./scripts/run_GDVB.sh configs/s2_mnist_idm.toml analyze --seed 10 --result_dir res_s1
./scripts/run_GDVB.sh configs/s2_mnist_idm.toml analyze --seed 11 --result_dir res_s1
./scripts/run_GDVB.sh configs/s2_mnist_idm.toml analyze --seed 12 --result_dir res_s1
./scripts/run_GDVB.sh configs/s2_mnist_idm.toml analyze --seed 13 --result_dir res_s1
./scripts/run_GDVB.sh configs/s2_mnist_idm.toml analyze --seed 14 --result_dir res_s1

# neu eps
./scripts/run_GDVB.sh configs/s2_mnist_eps.toml analyze --seed 10 --result_dir res_s1
./scripts/run_GDVB.sh configs/s2_mnist_eps.toml analyze --seed 11 --result_dir res_s1
./scripts/run_GDVB.sh configs/s2_mnist_eps.toml analyze --seed 12 --result_dir res_s1
./scripts/run_GDVB.sh configs/s2_mnist_eps.toml analyze --seed 13 --result_dir res_s1
./scripts/run_GDVB.sh configs/s2_mnist_eps.toml analyze --seed 14 --result_dir res_s1
