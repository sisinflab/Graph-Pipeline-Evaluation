from scipy.sparse import csr_matrix, save_npz
import pandas as pd

dataset = 'allrecipes'

train = pd.read_csv(f'data/{dataset}/train.tsv', sep='\t', header=None)
public_to_private_users = {u: idx for idx, u in enumerate(train[0].unique().tolist())}
public_to_private_items = {i: idx for idx, i in enumerate(train[1].unique().tolist())}
rows = [public_to_private_users[u] for u in train[0].tolist()]
cols = [public_to_private_items[i] for i in train[1].tolist()]
data = [1] * len(train)


sparse_train = csr_matrix((data, (rows, cols)), shape=(train[0].nunique(), train[1].nunique()))
save_npz(f'data/{dataset}/train_sparse.npz', sparse_train)
