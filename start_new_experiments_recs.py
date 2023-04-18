from elliot.run import run_experiment
import argparse

parser = argparse.ArgumentParser(description="Run sample main.")
parser.add_argument('--dataset', type=str, default='bookcrossing')
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

run_experiment(f"config_files/experiment_quartiles.yml",
               dataset=args.dataset,
               gpu=args.gpu,
               config_already_loaded=False)
