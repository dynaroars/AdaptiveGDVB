import argparse
import yaml
import os
import copy

CONFIG_DIR = 'exp_configs'

def generate(dataset):
    print('[+] Processing', dataset)
    dataset_config_dir = os.path.join(CONFIG_DIR, dataset)
    os.system(f'rm -rf {dataset_config_dir}')
    os.makedirs(dataset_config_dir, exist_ok=True)

    with open(f'{dataset}_default.yaml') as fp:
        default_config = yaml.safe_load(fp)
    print(default_config)

    BRANCH_MODES = ["approx", "ego"]
    SPLIT_ORDERS = ["ninf", "largest", "smallest"]
    APPROX_LEVELS = [1, 2, 3]
    
    idx = 0
    for branch in BRANCH_MODES:
        for split in SPLIT_ORDERS:
            for level in APPROX_LEVELS:
                new_config = copy.deepcopy(default_config)
                config_name = f'{branch}_{split}_{level}.yaml'
                print(idx, config_name)

                new_config['branch'] = branch
                new_config['split'] = split
                new_config['abstraction']['level'] = level

                idx += 1
                print(os.path.join(dataset_config_dir, config_name))
                with open(os.path.join(dataset_config_dir, config_name), 'w') as fp:
                    yaml.safe_dump(new_config, fp)

    print('[+] Total:', idx)

if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', type=str, required=True)
    args = parser.parse_args()   

    generate(args.benchmark)