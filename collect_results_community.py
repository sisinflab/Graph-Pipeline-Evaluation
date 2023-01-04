import os
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description="Run collect results community.")
parser.add_argument('--folder', type=str, default='allrecipes_results_tabusampler')
args = parser.parse_args()

file = os.listdir(f'./results/{args.folder}/')[0]
df_results = pd.read_csv(f'./results/{args.folder}/{file}', sep='\t')
df_results.insert(0, 'model_name', df_results['model'].apply(lambda row: row.split('_')[0]))
df_results.insert(1, 'model_params', df_results['model'].apply(lambda row: row.split(row.split('_')[0] + '_')[1]))
df_results.drop('model', axis=1, inplace=True)
df_results.reset_index(drop=True, inplace=True)

for name, group in df_results.groupby('model_name'):
    group.insert(0, 'model', group[['model_name', 'model_params']].apply(lambda x: "_".join(x), axis=1))
    group.drop(['model_name', 'model_params'], axis=1, inplace=True)
    group.to_csv(f'./results/{args.folder}/rec_{name}.tsv', sep='\t', index=None)
