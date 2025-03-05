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
    ATTACK = ['pgd', 'skip']
    BATCH = [1, max_batch]
    DEVICE = ['cpu', 'cuda']
    
    idx = 0
    for decision in DECISION:
        for abstraction in ABSTRACTION:
            for attack in ATTACK:
                for batch in BATCH:
                    for device in DEVICE:
                        new_config = copy.deepcopy(default_config)
                        config_name = f'exp{idx}_{device}_{batch}_{abstraction}_{decision}_{attack}.yaml'
                        print(idx, config_name)

                        new_config['general']['device'] = device
                        new_config['solver']['beta-crown']['batch_size'] = batch
                        new_config['bab']['branching']['method'] = decision

                        if attack == 'pgd':
                            new_config['attack']['pgd_order'] = 'before'
                        else:
                            new_config['attack']['pgd_order'] = 'skip'

                        if abstraction == 'beta-crown':
                            new_config['solver']['beta-crown']['beta'] = True
                        else:
                            new_config['solver']['beta-crown']['beta'] = False


                        idx += 1
                        print(os.path.join(dataset_config_dir, config_name))
                        with open(os.path.join(dataset_config_dir, config_name), 'w') as fp:
                            yaml.safe_dump(new_config, fp)

    print('[+] Total:', idx)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', type=str, required=True)
    parser.add_argument('--batch', type=int, default=200)
    args = parser.parse_args()   

    generate(args.benchmark, args.batch)