from elliot.run import run_experiment
from yaml import FullLoader as FullLoader
from yaml import load
import argparse

parser = argparse.ArgumentParser(description="Run sample main.")
parser.add_argument('--dataset', type=str, default='gowalla')
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

config_file = open(f"config_files/experiment_results_gfcf.yml")
config = load(config_file, Loader=FullLoader)

for idx, complex_metric in enumerate(config['experiment']['evaluation']['complex_metrics']):
    if complex_metric['metric'] in ['BiasDisparityBD', 'BiasDisparityBR', 'BiasDisparityBS']:
        config['experiment']['evaluation']['complex_metrics'][idx]['user_clustering_file'] = \
            config['experiment']['evaluation']['complex_metrics'][idx]['user_clustering_file'].format(args.dataset)
        config['experiment']['evaluation']['complex_metrics'][idx]['item_clustering_file'] = \
            config['experiment']['evaluation']['complex_metrics'][idx]['item_clustering_file'].format(args.dataset)
    else:
        config['experiment']['evaluation']['complex_metrics'][idx]['clustering_file'] = \
            config['experiment']['evaluation']['complex_metrics'][idx]['clustering_file'].format(args.dataset)

run_experiment(f"config_files/experiment.yml",
               dataset=args.dataset,
               gpu=args.gpu,
               config_already_loaded=True,
               config=config)
