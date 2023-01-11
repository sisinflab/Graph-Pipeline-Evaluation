import argparse
import itertools
from operator import itemgetter, add

import networkx
import pandas as pd
from networkx.algorithms import community
from networkx import bipartite

parser = argparse.ArgumentParser(description="Run community detection and related stats")
parser.add_argument('--dataset', type=str, default='allrecipes')
args = parser.parse_args()

train = pd.read_csv(f'./data/{args.dataset}/train.tsv', sep='\t', header=None)

users = list(range(train[0].nunique()))
items = list(range(train[1].nunique()))

public_to_private_users = {u: idx for idx, u in enumerate(train[0].unique())}
private_to_public_users = {idx: u for u, idx in public_to_private_users.items()}
public_to_private_items = {i: idx + len(users) for idx, i in enumerate(train[1].unique())}
private_to_public_items = {idx: i for i, idx in public_to_private_items.items()}

rows = [public_to_private_users[u] for u in train[0].tolist()]
cols = [public_to_private_items[i] for i in train[1].tolist()]

graph = networkx.Graph()
graph.add_nodes_from(users, bipartite='users')
graph.add_nodes_from(items, bipartite='items')
graph.add_edges_from(list(zip(rows, cols)))

users_nodes, items_nodes = bipartite.sets(graph)
user_user = bipartite.projected_graph(graph, users_nodes)
item_item = bipartite.projected_graph(graph, items_nodes)

user_communities = community.louvain_communities(user_user, seed=123)
item_communities = community.louvain_communities(item_item, seed=123)

user_communities_private = list(itertools.chain(*(zip(itemgetter(*u)(private_to_public_users), itertools.repeat(idx))
                                                  for idx, u in enumerate(user_communities))))
item_communities_private = list(itertools.chain(*(zip(itemgetter(*i)(private_to_public_items), itertools.repeat(idx))
                                                  for idx, i in enumerate(item_communities))))

user_communities_private = pd.DataFrame(user_communities_private).sort_values([0]).reset_index(drop=True)
item_communities_private = pd.DataFrame(item_communities_private).sort_values([0]).reset_index(drop=True)

train = pd.merge(train, user_communities_private, how='inner', left_on=0, right_on=0)
train = pd.merge(train, item_communities_private, how='inner', left_on='1_x', right_on=0)
train.drop(['0_y'], axis=1, inplace=True)

# users communities
stats_users = train.groupby(['1_y']).agg({2: 'count', '0_x': 'nunique', '1_x': 'nunique'})
stats_users['density'] = 1 - stats_users[2]/(stats_users['0_x']*stats_users['1_x'])
stats_users['mean'] = train.groupby(['1_y', '0_x'])[2].count().mean(level='1_y')
stats_users.reset_index(inplace=True)
stats_users.columns = ['community', 'ratings', 'users', 'items', 'density', 'mean']

# items communities
stats_items = train.groupby([1]).agg({2: 'count', '0_x': 'nunique', '1_x': 'nunique'})
stats_items['density'] = 1 - stats_items[2]/(stats_items['0_x']*stats_items['1_x'])
stats_items['mean'] = train.groupby([1, '1_x'])[2].count().mean(level=1)
stats_items.reset_index(inplace=True)
stats_items.columns = ['community', 'ratings', 'users', 'items', 'density', 'mean']

# saving files users
user_communities_private.to_csv(f'./data/{args.dataset}/users_communities.tsv', sep='\t', header=False, index=False)
stats_users.to_csv(f'./data/{args.dataset}/users_communities_stats.tsv', sep='\t', index=False)

# saving files users
item_communities_private.to_csv(f'./data/{args.dataset}/items_communities.tsv', sep='\t', header=False, index=False)
stats_items.to_csv(f'./data/{args.dataset}/items_communities_stats.tsv', sep='\t', index=False)
