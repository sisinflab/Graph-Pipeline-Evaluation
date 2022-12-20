import pandas as pd

dataset = 'bookcrossing'

projected = ['QUBOBipartiteCommunityDetection', 'QUBOBipartiteProjectedCommunityDetection']
strategies = ['SimulatedAnnealingSampler', 'SteepestDescentSolver', 'TabuSampler']

train = pd.read_csv(f'data/{dataset}/train.tsv', sep='\t', header=None)

private_to_public_users = {idx: u for idx, u in enumerate(train[0].unique().tolist())}
private_to_public_items = {idx: i for idx, i in enumerate(train[1].unique().tolist())}


def to_public_users(idx):
    return private_to_public_users[idx]


def to_public_items(idx):
    return private_to_public_items[idx]


for p in projected:
    for s in strategies:
        user_groups_community = pd.read_csv(f'data/{dataset}/{p}/{s}/user_groups_community.tsv', sep='\t', header=None)
        item_groups_community = pd.read_csv(f'data/{dataset}/{p}/{s}/item_groups_community.tsv', sep='\t', header=None)
        user_groups_community[0] = user_groups_community[0].apply(to_public_users)
        item_groups_community[0] = item_groups_community[0].apply(to_public_items)
        user_groups_community.to_csv(f'data/{dataset}/{p}/{s}/user_groups_community.tsv', sep='\t',
                                     header=None, index=None)
        item_groups_community.to_csv(f'data/{dataset}/{p}/{s}/item_groups_community.tsv', sep='\t',
                                     header=None, index=None)
