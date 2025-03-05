import argparse
import yaml
import os
import copy

CONFIG_DIR = 'exp_configs'

def generate(dataset, max_batch):
    print('[+] Processing', dataset)
    dataset_config_dir = os.path.join(CONFIG_DIR, dataset)
    os.system(f'rm -rf {dataset_config_dir}')
    os.makedirs(dataset_config_dir, exist_ok=True)

    with open(f'{dataset}_default.yaml') as fp:
        default_config = yaml.safe_load(fp)
    print(default_config)

    DECISION = ["babsr", "fsb", "kfsb"]
    ABSTRACTION = ["alpha-crown", "beta-crown"]
    ATTACK = ['pgd']
    BATCH = [10, max_batch]
    DEVICE = ['cuda', 'cpu']
    
    idx = 0
    for decision in DECISION:
        for abstraction in ABSTRACTION:
            for attack in ATTACK:
                for batch in BATCH:
                    for device in DEVICE:
                        new_config = copy.deepcopy(default_config)
                        config_name = f'{device}_{batch}_{abstraction}_{decision}_{attack}.yaml'
                        print(idx, config_name)

                        new_config['general']['device'] = device
                        new_config['solver']['batch_size'] = batch
                        new_config['bab']['branching']['method'] = decision

                        if abstraction == 'alpha-crown':
                            new_config['solver']['beta-crown']['beta'] = False
                        elif abstraction == 'beta-crown':
                            new_config['solver']['beta-crown']['beta'] = True
                        else:
                            raise NotImplementedError


                        if attack == 'pgd':
                            new_config['attack']['pgd_order'] = 'before'
                        elif attack == 'bab':
                            new_config['bab']['attack']['enabled'] = True
                            new_config['bab']['get_upper_bound'] = True
                            new_config['general']['enable_incomplete_verification'] = True
                        else:
                            raise NotImplementedError

                        idx += 1
                        print(os.path.join(dataset_config_dir, config_name))
                        with open(os.path.join(dataset_config_dir, config_name), 'w') as fp:
                            yaml.safe_dump(new_config, fp)

    print('[+] Total:', idx)

if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', type=str, required=True)
    parser.add_argument('--batch', type=int, default=1000)
    args = parser.parse_args()   

    generate(args.benchmark, args.batch)
    
    # mnistfc_medium_gdvb_all