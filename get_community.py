import networkx
import pandas as pd
from networkx.algorithms import community
from networkx import bipartite

dataset = 'allrecipes'

train = pd.read_csv(f'./data/{dataset}/train.tsv', sep='\t', header=None)

users = list(range(train[0].nunique()))
items = list(range(train[0].nunique(), train[0].nunique() + train[1].nunique()))

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

user_user = bipartite.projected_graph(graph, users)
item_item = bipartite.projected_graph(graph, items)

user_communities = community.louvain_communities(user_user)
item_communities = community.louvain_communities(item_item)

# https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.modularity_max.greedy_modularity_communities.html#networkx.algorithms.community.modularity_max.greedy_modularity_communities
