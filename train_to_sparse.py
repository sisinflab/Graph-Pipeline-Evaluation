from scipy.sparse import csr_matrix, save_npz
import pandas as pd
import json

dataset = 'clothing'

train = pd.read_csv(f'data/{dataset}/train.tsv', sep='\t', header=None)
public_to_private_users = {u: idx for idx, u in enumerate(train[0].unique().tolist())}
public_to_private_items = {i: idx for idx, i in enumerate(train[1].unique().tolist())}
private_to_public_users = {idx: u for u, idx in public_to_private_users.items()}
private_to_public_items = {idx: i for i, idx in public_to_private_items.items()}
rows = [public_to_private_users[u] for u in train[0].tolist()]
cols = [public_to_private_items[i] for i in train[1].tolist()]
data = [1] * len(train)


sparse_train = csr_matrix((data, (rows, cols)), shape=(train[0].nunique(), train[1].nunique()))
save_npz(f'data/{dataset}/train_sparse.npz', sparse_train)

json_private_to_public_users = json.dumps(private_to_public_users, indent=4)
json_private_to_public_items = json.dumps(private_to_public_items, indent=4)

with open(f'data/{dataset}/private_to_public_users.json', "w") as outfile:
    outfile.write(json_private_to_public_users)

with open(f'data/{dataset}/private_to_public_items.json', "w") as outfile:
    outfile.write(json_private_to_public_items)
