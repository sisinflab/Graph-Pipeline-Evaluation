from elliot.run import run_experiment
import argparse
import pandas as pd

from yaml import FullLoader as FullLoader
from yaml import load

import copy

parser = argparse.ArgumentParser(description="Run sample main.")
parser.add_argument('--dataset', type=str, default='allrecipes')
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

df = pd.read_csv(f'results/{args.dataset}/performance/best_iterations.tsv', header=None, sep='\t')

config_file = open(f"config_files/experiment_results.yml")
original_config = load(config_file, Loader=FullLoader)

models_params = {
    'NGCF': {
        'lr': ['lr', float],
        'e': ['epochs', int],
        'factors': ['factors', int],
        'bs': ['batch_size', int],
        'l_w': ['l_w', float],
        'n_layers': ['n_layers', int],
        'weight_size': ['weight_size', int],
        'node_dropout': ['node_dropout', float],
        'message_dropout': ['message_dropout', float],
        'normalize': ['normalize', bool],
        'seed': ['seed', int]
    },
    'LightGCN': {
        'lr': ['lr', float],
        'e': ['epochs', int],
        'factors': ['factors', int],
        'bs': ['batch_size', int],
        'l_w': ['l_w', float],
        'n_layers': ['n_layers', int],
        'normalize': ['normalize', bool],
        'seed': ['seed', int]
    },
    'DGCF': {
        'lr': ['lr', float],
        'e': ['epochs', int],
        'factors': ['factors', int],
        'bs': ['batch_size', int],
        'l_w_bpr': ['l_w_bpr', float],
        'l_w_ind': ['l_w_ind', float],
        'ind_batch_size': ['ind_batch_size', int],
        'n_layers': ['n_layers', int],
        'routing_iterations': ['routing_iterations', int],
        'intents': ['intents', int],
        'seed': ['seed', int]
    },
    'LRGCCF': {
        'lr': ['lr', float],
        'e': ['epochs', int],
        'factors': ['factors', int],
        'bs': ['batch_size', int],
        'l_w': ['l_w', float],
        'n_layers': ['n_layers', int],
        'normalize': ['normalize', bool],
        'seed': ['seed', int]
    },
    'SGL': {
        'lr': ['lr', float],
        'e': ['epochs', int],
        'factors': ['factors', int],
        'bs': ['batch_size', int],
        'l_w': ['l_w', float],
        'n_layers': ['n_layers', int],
        'ssl_temp': ['ssl_temp', float],
        'ssl_reg': ['ssl_reg', float],
        'ssl_ratio': ['ssl_ratio', float],
        'sampling': ['sampling', str]
    },
    'UltraGCN': {
        'lr': ['lr', float],
        'e': ['epochs', int],
        'factors': ['factors', int],
        'bs': ['batch_size', int],
        'g': ['g', float],
        'l': ['l', float],
        'w1': ['w1', float],
        'w2': ['w2', float],
        'w3': ['w3', float],
        'w4': ['w4', float],
        'ii_n_n': ['ii_n_n', int],
        'n_n_w': ['n_n_w', int],
        's_s_p': ['s_s_p', bool],
        'i_w': ['i_w', float],
        'seed': ['seed', int]
    },
    'SVDGCN': {
        'factors': ['factors', int],
        'e': ['epochs', int],
        'bs': ['batch_size', int],
        'l_w': ['l_w', float],
        'lr': ['lr', float],
        'req_vec': ['req_vec', int],
        'beta': ['beta', float],
        'alpha': ['alpha', float],
        'coef_u': ['coef_u', float],
        'coef_i': ['coef_i', float],
        'seed': ['seed', int]
    }
}

for ind in df.index:
    current_config = df[0][ind]
    best_iteration = int(df[1][ind])
    model = current_config.split('_')[0]
    config = copy.deepcopy(original_config)
    for idx, complex_metric in enumerate(config['experiment']['evaluation']['complex_metrics']):
        if complex_metric['metric'] in ['BiasDisparityBD', 'BiasDisparityBR', 'BiasDisparityBS']:
            config['experiment']['evaluation']['complex_metrics'][idx]['user_clustering_file'] = \
                config['experiment']['evaluation']['complex_metrics'][idx]['user_clustering_file'].format(args.dataset)
            config['experiment']['evaluation']['complex_metrics'][idx]['item_clustering_file'] = \
                config['experiment']['evaluation']['complex_metrics'][idx]['item_clustering_file'].format(args.dataset)
        else:
            config['experiment']['evaluation']['complex_metrics'][idx]['clustering_file'] = \
                config['experiment']['evaluation']['complex_metrics'][idx]['clustering_file'].format(args.dataset)
    config["experiment"]["models"][f"external.{model}"] = config["experiment"]["models"].pop("external.ModelName")
    config["experiment"]["models"][f"external.{model}"]["meta"]["validation_rate"] = best_iteration
    current_model_params = models_params[model]
    for key, value in current_model_params.items():
        parameter = current_config.split(key + '=')[1].split("_")[0]
        if value[1] == bool:
            parameter = True if parameter == 'True' else False
        elif '$' in parameter:
            parameter = parameter.replace('$', '.')
        parameter = value[1](parameter)
        if key == 'e':
            config["experiment"]["models"][f"external.{model}"][value[0]] = best_iteration
        else:
            config["experiment"]["models"][f"external.{model}"][value[0]] = parameter
    run_experiment(f"config_files/experiment.yml",
                   dataset=args.dataset,
                   gpu=args.gpu,
                   config_already_loaded=True,
                   config=config)
