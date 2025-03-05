import argparse
import os
import copy
import json

CONFIG_DIR = 'exp_configs'

def generate(dataset, max_batch):
    print('[+] Processing', dataset)
    dataset_config_dir = os.path.join(CONFIG_DIR, dataset)
    os.system(f'rm -rf {dataset_config_dir}')
    os.makedirs(dataset_config_dir, exist_ok=True)

    with open(f'{dataset}_default.json') as fp:
        default_config = json.load(fp)

    print(default_config)

    DECISION = ['babsr', 'filtered_smart_branching', 'active_constraint_score']
    ABSTRACTION = ["alpha-crown", "prima"]
    ATTACK = ['pgd']
    BATCH = [1, max_batch]
    DEVICE = ['cuda']
    
    idx = 0
    for decision in DECISION:
        for abstraction in ABSTRACTION:
            for attack in ATTACK:
                for batch in BATCH:
                    for device in DEVICE:
                        new_config = copy.deepcopy(default_config)
                        config_name = f'{device}_{batch}_{abstraction}_{decision}_{attack}.json'
                        print(idx, config_name)
                        idx += 1

                        new_config['use_gpu'] = device == 'cuda'
                        new_config['branching']['method'] = decision
                        new_config['bab_batch_size'] = [batch] * 6
                        new_config['optimize_alpha'] = True
                        new_config['optimize_prima'] = abstraction == 'prima'

                        # print(os.path.join(dataset_config_dir, config_name))
                        with open(os.path.join(dataset_config_dir, config_name), 'w') as fp:
                            json.dump(new_config, fp, indent=2)

    print('[+] Total:', idx)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', type=str, required=True)
    parser.add_argument('--batch', type=int, default=8)
    args = parser.parse_args()   

    generate(args.benchmark, args.batch)