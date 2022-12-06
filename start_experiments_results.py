from elliot.run import run_experiment
import argparse
import pandas as pd

from yaml import FullLoader as FullLoader
from yaml import load

parser = argparse.ArgumentParser(description="Run sample main.")
parser.add_argument('--dataset', type=str, default='allrecipes')
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

df = pd.read_csv(f'results/{args.dataset}/performance/best_iterations.tsv', header=None, sep='\t')

config_file = open(f"config_files/experiment_results.yml")
config = load(config_file, Loader=FullLoader)

for ind in df.index:
    current_config = df[0][ind]
    best_iteration = df[1][ind]
    model = current_config.split('_')[0]
    print()
    run_experiment(f"config_files/experiment.yml",
                   dataset=args.dataset,
                   gpu=args.gpu,
                   config_already_loaded=True,
                   config=config)
