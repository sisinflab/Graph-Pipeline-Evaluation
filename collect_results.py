import os
import re
import argparse
import pandas as pd

from yaml import FullLoader as FullLoader
from yaml import load

parser = argparse.ArgumentParser(description="Run collect results.")
parser.add_argument('--folder', type=str, default='bookcrossing_results')
args = parser.parse_args()

all_configurations_file = open(f"results/all_configurations.yml")
all_configurations = load(all_configurations_file, Loader=FullLoader)

result_files = [f for f in os.listdir(f'./results/{args.folder}') if f != 'rec_GFCF.tsv']

df_results = pd.DataFrame()

for file in result_files:
    current_df = pd.read_csv(f'./results/{args.folder}/{file}', sep='\t')
    current_df['filename'] = file
    if len(current_df.columns) < 35:  # not all metrics have been calculated
        print(file)
    df_results = pd.concat([df_results, current_df], axis=0)

df_results.insert(0, 'model_name', df_results['model'].apply(lambda row: row.split('_')[0]))
df_results.insert(1, 'model_params', df_results['model'].apply(lambda row: row.split(row.split('_')[0] + '_')[1]))
df_results.drop('model', axis=1, inplace=True)
df_results.reset_index(drop=True, inplace=True)

# df_results[df_results['model_name'] == 'NGCF'][df_results[df_results['model_name'] == 'NGCF']['model_params'].duplicated(keep=False)]['filename'].tolist()

missing_configs = False

for name, group in df_results.groupby('model_name'):
    print(f'Model name: {name}')
    print(f'Number of retrieved configurations: {len(group)}')
    print(f'Missing configurations: {len(all_configurations[name]) - len(group)}')
    if len(all_configurations[name]) - len(group) > 0:
        missing_configs = True
        current_model_params = [re.sub(r'seed=123_e=\d+_', '', c_m_p) for c_m_p in group['model_params'].tolist()]
        missing_configurations = set(all_configurations[name]).difference(set(current_model_params))
        for m_c in missing_configurations:
            print(m_c)
    print('\n')

if not missing_configs:
    for name, group in df_results.groupby('model_name'):
        group.insert(0, 'model', group[['model_name', 'model_params']].apply(lambda x: "_".join(x), axis=1))
        group.drop(['model_name', 'model_params'], axis=1, inplace=True)
        group.to_csv(f'./results/{args.folder}/rec_{name}.tsv', sep='\t', index=None)

