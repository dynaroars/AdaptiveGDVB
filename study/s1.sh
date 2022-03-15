#!/usr/bin/env bash

# mnist_3x1024
./scripts/run_GDVB.sh configs/s1_mnist_3x1024.toml analyze --seed 10 --result_dir res_s1
./scripts/run_GDVB.sh configs/s1_mnist_3x1024.toml analyze --seed 11 --result_dir res_s1
./scripts/run_GDVB.sh configs/s1_mnist_3x1024.toml analyze --seed 12 --result_dir res_s1
./scripts/run_GDVB.sh configs/s1_mnist_3x1024.toml analyze --seed 13 --result_dir res_s1
./scripts/run_GDVB.sh configs/s1_mnist_3x1024.toml analyze --seed 14 --result_dir res_s1

# mnist_conv_big
./scripts/run_GDVB.sh configs/s1_mnist_convbig.toml analyze --seed 10 --result_dir res_s1
./scripts/run_GDVB.sh configs/s1_mnist_convbig.toml analyze --seed 11 --result_dir res_s1
./scripts/run_GDVB.sh configs/s1_mnist_convbig.toml analyze --seed 12 --result_dir res_s1
./scripts/run_GDVB.sh configs/s1_mnist_convbig.toml analyze --seed 13 --result_dir res_s1
./scripts/run_GDVB.sh configs/s1_mnist_convbig.toml analyze --seed 14 --result_dir res_s1

# cifar_conv_big
./scripts/run_GDVB.sh configs/s1_cifar_convbig.toml analyze --seed 10 --result_dir res_s1
./scripts/run_GDVB.sh configs/s1_cifar_convbig.toml analyze --seed 11 --result_dir res_s1
./scripts/run_GDVB.sh configs/s1_cifar_convbig.toml analyze --seed 12 --result_dir res_s1
./scripts/run_GDVB.sh configs/s1_cifar_convbig.toml analyze --seed 13 --result_dir res_s1
./scripts/run_GDVB.sh configs/s1_cifar_convbig.toml analyze --seed 14 --result_dir res_s1

# dave
./scripts/run_GDVB.sh configs/s1_dave.toml analyze --seed 10 --result_dir res_s1
./scripts/run_GDVB.sh configs/s1_dave.toml analyze --seed 11 --result_dir res_s1
./scripts/run_GDVB.sh configs/s1_dave.toml analyze --seed 12 --result_dir res_s1
./scripts/run_GDVB.sh configs/s1_dave.toml analyze --seed 13 --result_dir res_s1
./scripts/run_GDVB.sh configs/s1_dave.toml analyze --seed 14 --result_dir res_s1