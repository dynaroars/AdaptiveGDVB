import argparse
from email.policy import default
from pyfiglet import Figlet


def parse_args():
    f = Figlet(font='slant')
    print(f.renderText('EvoGDVB'), end='')

    parser = argparse.ArgumentParser(
        description='Evolutionary Generative Diverse DNN Verification Benchmarks',
        prog='EvoGDVB')

    parser.add_argument('configs', type=str,
                        help='Configurations file.')
    parser.add_argument('task', type=str,
                        choices=['evolutionary'],
                        help='Select tasks to perform.')
    parser.add_argument('--seed', type=int,
                        default=0,
                        help='Random seed.')
    parser.add_argument('--result_dir', type=str,
                        default='./results/',
                        help='Root directory.')
    parser.add_argument('--platform', type=str,
                        default='local',
                        choices=['local', 'slurm'],
                        help='Platform to run jobs.')
    parser.add_argument('--override', action='store_true',
                        help='Override existing logs.')
    parser.add_argument('--debug', action='store_true',
                        help='Print debug log.')
    parser.add_argument('--dumb', action='store_true',
                        help='Silent mode.')

    return parser.parse_args()
