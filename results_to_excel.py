import argparse
import re
import pandas as pd

parser = argparse.ArgumentParser(description="Run results to excel.")
parser.add_argument('--folder', type=str, default='bookcrossing_results_community')
args = parser.parse_args()

models_params = {
    'NGCF': {
        'lr': ['lr', float],
        'factors': ['factors', int],
        'l_w': ['l_w', float],
        'n_layers': ['n_layers', int],
        'weight_size': ['weight_size', int],
        'node_dropout': ['node_dropout', float],
        'message_dropout': ['message_dropout', float],
        'normalize': ['normalize', bool]
    },
    'LightGCN': {
        'lr': ['lr', float],
        'factors': ['factors', int],
        'l_w': ['l_w', float],
        'n_layers': ['n_layers', int],
        'normalize': ['normalize', bool]
    },
    'DGCF': {
        'lr': ['lr', float],
        'factors': ['factors', int],
        'l_w_bpr': ['l_w_bpr', float],
        'l_w_ind': ['l_w_ind', float],
        'ind_batch_size': ['ind_batch_size', int],
        'n_layers': ['n_layers', int],
        'routing_iterations': ['routing_iterations', int],
        'intents': ['intents', int]
    },
    'LRGCCF': {
        'lr': ['lr', float],
        'factors': ['factors', int],
        'l_w': ['l_w', float],
        'n_layers': ['n_layers', int],
        'normalize': ['normalize', bool]
    },
    'SGL': {
        'lr': ['lr', float],
        'factors': ['factors', int],
        'l_w': ['l_w', float],
        'n_layers': ['n_layers', int],
        'ssl_temp': ['ssl_temp', float],
        'ssl_reg': ['ssl_reg', float],
        'ssl_ratio': ['ssl_ratio', float],
        'sampling': ['sampling', str]
    },
    'UltraGCN': {
        'lr': ['lr', float],
        'factors': ['factors', int],
        'g': ['g', float],
        'l': ['l', float],
        'w1': ['w1', float],
        'w2': ['w2', float],
        'w3': ['w3', float],
        'w4': ['w4', float],
        'ii_n_n': ['ii_n_n', int],
        'n_n_w': ['n_n_w', int],
        's_s_p': ['s_s_p', bool],
        'i_w': ['i_w', float]
    },
    'SVDGCN': {
        'factors': ['factors', int],
        'l_w': ['l_w', float],
        'lr': ['lr', float],
        'req_vec': ['req_vec', int],
        'beta': ['beta', float],
        'alpha': ['alpha', float],
        'coef_u': ['coef_u', float],
        'coef_i': ['coef_i', float]
    },
    'GFCF': {
        'svd_factors': ['svd_factors', int],
        'alpha': ['alpha', float]
    }
}

files = ['rec_DGCF_community.tsv',
         'rec_GFCF_community.tsv',
         'rec_LightGCN_community.tsv',
         'rec_LRGCCF_community.tsv',
         'rec_NGCF_community.tsv',
         'rec_SGL_community.tsv',
         'rec_SVDGCN_community.tsv',
         'rec_UltraGCN_community.tsv']

for f in files:
    model = f.split('_')[1].split('.')[0]
    df = pd.read_csv(f'results/{args.folder}/{f}', sep='\t')
    df.insert(0, 'model_name', df['model'].apply(lambda row: row.split('_')[0]))
    df.insert(1, 'model_params', df['model'].apply(lambda row: row.split(row.split('_')[0] + '_')[1]))
    df.drop('model', axis=1, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['model_params'] = df['model_params'].apply(lambda row: re.sub(r'seed=123_e=\d+_bs=\d+_', '', row))
    if 'GFCF' in f:
        df['model_params'] = df['model_params'].apply(lambda row: re.sub(r'batch_eval=-\d+_', '', row))
    if 'DGCF' in f:
        df['model_params'] = df['model_params'].apply(lambda row: re.sub(r'ind_batch_size=-\d+_', '', row))
    for k in models_params[model].keys():
        values = [mp.split(k + '=')[1].split("_")[0] for mp in df['model_params'].tolist()]
        if models_params[model][k][1] == float:
            values = [mp.replace('$', '.') if '$' in mp else mp for mp in values]
        elif models_params[model][k][1] == bool:
            values = [True if mp == 'True' else False for mp in values]
        values = [models_params[model][k][1](mp) for mp in values]
        df.insert(2, k, values, True)
    df.drop('model_params', axis=1, inplace=True)
    df.to_excel(f'results/{args.folder}/{f.split(".")[0]}_params.xlsx', index=None)
