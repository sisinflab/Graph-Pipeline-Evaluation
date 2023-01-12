from elliot.run import run_experiment
import argparse

parser = argparse.ArgumentParser(description="Run sample main.")
parser.add_argument('--dataset', type=str, default='bookcrossing_test_clustered_metrics')
parser.add_argument('--gpu', type=int, default=-1)
args = parser.parse_args()

run_experiment(f"config_files/internal_test_clustered_metrics.yml",
               dataset=args.dataset,
               gpu=args.gpu,
               config_already_loaded=False)