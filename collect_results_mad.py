import argparse
import pandas as pd
import re

parser = argparse.ArgumentParser(description="Run collect results mad.")
parser.add_argument('--folder', type=str, default='allrecipes_results')
parser.add_argument('--file', type=str, default='rec_cutoff_20_relthreshold_0_2022_12_29_22_23_36.tsv')
args = parser.parse_args()

df_results = pd.read_csv(f'./results/{args.folder}/{args.file}', sep='\t')
df_results.drop(['nDCG', 'Recall', 'Precision', 'HR',
                 'MAR', 'MAP', 'MRR', 'F1', 'ACLT', 'APLT', 'ARP', 'PopREO',
                 'ItemCoverage', 'Gini'], axis=1, inplace=True)
df_results.reset_index(drop=True, inplace=True)

model_results = dict()

for idx, row in df_results.iterrows():
    model = row['model'].split('_')[0]
    if model not in model_results:
        model_results[model] = pd.read_csv(f'./results/{args.folder}/rec_{model}.tsv', sep='\t')
        model_results[model]['UserMADrating_WarmColdUsers'] = pd.Series([0.0] * len(model_results[model]))
        model_results[model]['ItemMADrating_WarmColdItems'] = pd.Series([0.0] * len(model_results[model]))
    model_results[model].loc[
        model_results[model]['model'] == re.sub(r'_it=\d+', '', row['model']), 'UserMADrating_WarmColdUsers'] = \
        row['UserMADrating_WarmColdUsers']
    model_results[model].loc[
        model_results[model]['model'] == re.sub(r'_it=\d+', '', row['model']), 'ItemMADrating_WarmColdItems'] = \
        row['ItemMADrating_WarmColdItems']

for key, value in model_results.items():
    if 'filename' in value.columns:
        value.drop('filename', axis=1, inplace=True)
    value.to_csv(f'./results/{args.folder}/rec_{key}.tsv', sep='\t', index=None)

