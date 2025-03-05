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

    SNC_MODES = [True, False]
    TIGHTENING_STRATEGIES = ["deeppoly", "sbt", "none"]
    BRANCH_MODES = ["pseudo-impact", "polarity", "largest-interval"] #, "relu-violation"]
    VERBOSITY = 2
    NUM_WORKERS = 64
    
    idx = 0
    for branch in BRANCH_MODES:
        for tight in TIGHTENING_STRATEGIES:
            for snc in SNC_MODES:
                new_config = copy.deepcopy(default_config)
                config_name = f'{branch}_{tight}_{snc}.yaml'
                print(idx, config_name)

                new_config['branch'] = branch
                new_config['tightening_strategy'] = tight
                new_config['snc'] = snc
                new_config['num_workers'] = NUM_WORKERS
                new_config['verbosity'] = VERBOSITY

                idx += 1
                print(os.path.join(dataset_config_dir, config_name))
                with open(os.path.join(dataset_config_dir, config_name), 'w') as fp:
                    yaml.safe_dump(new_config, fp)

    print('[+] Total:', idx)

if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', type=str, default="ablation_mnistfc_tiny")
    args = parser.parse_args()   

    generate(args.benchmark)