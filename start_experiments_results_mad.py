from elliot.run import run_experiment
import argparse

from yaml import FullLoader as FullLoader
from yaml import load

parser = argparse.ArgumentParser(description="Run sample main.")
parser.add_argument('--dataset', type=str, default='allrecipes')

args = parser.parse_args()

config_file = open(f"config_files/experiment_results.yml")
config = load(config_file, Loader=FullLoader)

for idx, complex_metric in enumerate(config['experiment']['evaluation']['complex_metrics']):
    config['experiment']['evaluation']['complex_metrics'][idx]['clustering_file'] = \
            config['experiment']['evaluation']['complex_metrics'][idx]['clustering_file'].format(
                args.dataset)

config["experiment"]["models"]["RecommendationFolder"]["folder"] = \
    config["experiment"]["models"]["RecommendationFolder"]["folder"].format(args.dataset)

run_experiment(f"config_files/experiment.yml",
               dataset=args.dataset,
               gpu=-1,
               config_already_loaded=True,
               config=config)
