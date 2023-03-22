import torch.utils.data as data
import numpy as np
import random
from torch.utils.data import DataLoader
import torch


class BPRData(data.Dataset):
    def __init__(self,
                 train_dict=None,
                 num_item=0,
                 num_ng=1,
                 seed=42,
                 batch_size=1,
                 data_set_count=0):
        super(BPRData, self).__init__()

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

        self.num_item = num_item
        self.train_dict = train_dict
        self.num_ng = num_ng
        self.set_all_item = list(range(num_item))
        self.features_fill = []
        self.data_set_count = data_set_count
        self.train_loader = DataLoader(self, batch_size=batch_size, shuffle=True)

    def ng_sample(self):
        for user_id in self.train_dict:
            positive_list = list(self.train_dict[user_id].keys())
            for item_i in positive_list:
                for t in range(self.num_ng):
                    item_j = np.random.randint(self.num_item)
                    while item_j in positive_list:
                        item_j = np.random.randint(self.num_item)
                    self.features_fill.append([user_id, item_i, item_j])

    def __getitem__(self, idx):
        features = self.features_fill

        user = features[idx][0]
        item_i = features[idx][1]
        item_j = features[idx][2]
        return user, item_i, item_j

    def __len__(self):
        return self.num_ng * self.data_set_count
