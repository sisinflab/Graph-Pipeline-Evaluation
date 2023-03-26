"""
Module description:

"""

__version__ = '0.3.0'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it'

from abc import ABC

import torch
import numpy as np
import random


class DGCFModel(torch.nn.Module, ABC):
    def __init__(self,
                 num_users,
                 num_items,
                 learning_rate,
                 embed_k,
                 l_w_bpr,
                 l_w_ind,
                 n_layers,
                 intents,
                 routing_iterations,
                 edge_index,
                 random_seed,
                 pick=0,
                 name="DGCF",
                 **kwargs
                 ):
        super().__init__()

        # set seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_users = num_users
        self.num_items = num_items
        self.embed_k = embed_k
        self.learning_rate = learning_rate
        self.l_w_bpr = l_w_bpr
        self.l_w_ind = l_w_ind
        self.n_layers = n_layers
        self.intents = intents
        self.routing_iterations = routing_iterations
        self.edge_index = torch.tensor(edge_index, dtype=torch.int64)
        self.all_h_list = edge_index[0].tolist()
        self.all_t_list = edge_index[1].tolist()

        initializer = torch.nn.init.xavier_uniform_
        self.Gu = torch.nn.Parameter(initializer(torch.empty(self.num_users, self.embed_k)))
        self.Gu.to(self.device)
        self.Gi = torch.nn.Parameter(initializer(torch.empty(self.num_items, self.embed_k)))
        self.Gi.to(self.device)

        self.pick_level = 1e10
        self.A_in_shape = (self.num_users + self.num_items, self.num_users + self.num_items)
        if pick == 1:
            self.is_pick = True
        else:
            self.is_pick = False

        self.softplus = torch.nn.Softplus()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def _convert_A_values_to_A_factors_with_P(self, f_num, A_factor_values, pick=True):
        A_factors = []
        D_col_factors = []
        D_row_factors = []
        # get the indices of adjacency matrix
        A_indices = np.mat([self.all_h_list, self.all_t_list]).transpose()
        D_indices = np.mat(
            [list(range(self.num_users + self.num_items)), list(range(self.num_users + self.num_items))]).transpose()

        # apply factor-aware softmax function over the values of adjacency matrix
        # ....A_factor_values is [n_factors, all_h_list]
        if pick:
            A_factor_scores = torch.nn.functional.softmax(A_factor_values, 0)
            min_A = torch.min(A_factor_scores, 0)
            index = A_factor_scores > (min_A.values + 0.0000001)
            index = index.type(torch.float32) * (
                    self.pick_level - 1.0) + 1.0  # adjust the weight of the minimum factor to 1/self.pick_level

            A_factor_scores = A_factor_scores * index
            A_factor_scores = A_factor_scores / torch.sum(A_factor_scores, 0)
        else:
            A_factor_scores = torch.nn.functional.softmax(A_factor_values, 0)

        for i in range(0, f_num):
            # in the i-th factor, couple the adjcency values with the adjacency indices
            # .... A i-tensor is a sparse tensor with size of [n_users+n_items,n_users+n_items]
            A_i_scores = A_factor_scores[i]
            A_i_tensor = torch.sparse_coo_tensor(np.array(A_indices).reshape(A_indices.shape[1], A_indices.shape[0]),
                                                 A_i_scores, self.A_in_shape)

            # get the degree values of A_i_tensor
            # .... D_i_scores_col is [n_users+n_items, 1]
            # .... D_i_scores_row is [1, n_users+n_items]
            D_i_col_scores = 1 / torch.sqrt(torch.sparse.sum(A_i_tensor, dim=1).to_dense())
            D_i_row_scores = 1 / torch.sqrt(torch.sparse.sum(A_i_tensor, dim=0).to_dense())

            # couple the laplacian values with the adjacency indices
            # .... A_i_tensor is a sparse tensor with size of [n_users+n_items, n_users+n_items]
            D_i_col_tensor = torch.sparse_coo_tensor(
                np.array(D_indices).reshape(D_indices.shape[1], D_indices.shape[0]), D_i_col_scores, self.A_in_shape)
            D_i_row_tensor = torch.sparse_coo_tensor(
                np.array(D_indices).reshape(D_indices.shape[1], D_indices.shape[0]), D_i_row_scores, self.A_in_shape)

            A_factors.append(A_i_tensor)
            D_col_factors.append(D_i_col_tensor)
            D_row_factors.append(D_i_row_tensor)

        # return a (n_factors)-length list of laplacian matrix
        return A_factors, D_col_factors, D_row_factors

    def propagate_embeddings(self, pick_=False):
        p_test = False
        p_train = False

        A_values = torch.ones(size=(self.intents, self.edge_index.shape[1]))

        ego_embeddings = torch.cat((self.Gu.to(self.device), self.Gi.to(self.device)), 0)
        all_embeddings = [ego_embeddings]

        output_factors_distribution = []

        for layer in range(self.n_layers):
            layer_embeddings = []
            ego_layer_embeddings = torch.split(ego_embeddings, self.embed_k // self.intents, 1)

            for t in range(self.routing_iterations):
                iter_embeddings = []
                A_iter_values = []

                if t == self.routing_iterations - 1:
                    p_test = pick_
                    p_train = False

                A_factors, D_col_factors, D_row_factors = self._convert_A_values_to_A_factors_with_P(self.intents,
                                                                                                     A_values,
                                                                                                     pick=p_train)

                for i in range(0, self.intents):
                    factor_embeddings = torch.sparse.mm(D_col_factors[i].to(self.device), ego_layer_embeddings[i].to(self.device))

                    factor_embeddings = torch.sparse.mm(A_factors[i].to(self.device), factor_embeddings.to(self.device))

                    factor_embeddings = torch.sparse.mm(D_col_factors[i].to(self.device), factor_embeddings.to(self.device))

                    iter_embeddings.append(factor_embeddings.to(self.device))

                    if t == self.routing_iterations - 1:
                        layer_embeddings = iter_embeddings

                    # get the factor-wise embeddings
                    # .... head_factor_embeddings is a dense tensor with the size of [all_h_list, embed_size/n_factors]
                    # .... analogous to tail_factor_embeddings
                    head_factor_embedings = factor_embeddings[self.all_h_list].to(self.device)
                    tail_factor_embedings = ego_layer_embeddings[i][self.all_t_list].to(self.device)

                    # .... constrain the vector length
                    # .... make the following attentive weights within the range of (0,1)
                    head_factor_embedings = torch.nn.functional.normalize(head_factor_embedings.to(self.device), dim=1)
                    tail_factor_embedings = torch.nn.functional.normalize(tail_factor_embedings.to(self.device), dim=1)

                    # get the attentive weights
                    # .... A_factor_values is a dense tensor with the size of [all_h_list,1]
                    A_factor_values = torch.sum(torch.mul(head_factor_embedings.to(self.device), torch.tanh(tail_factor_embedings.to(self.device))),
                                                dim=1)

                    # update the attentive weights
                    A_iter_values.append(A_factor_values.to(self.device))

                # pack (n_factors) adjacency values into one [n_factors, all_h_list] tensor
                A_iter_values = torch.stack(A_iter_values, 0)
                # add all layer-wise attentive weights up.
                A_values = A_values.to(self.device) + A_iter_values.to(self.device)

                if t == self.routing_iterations - 1:
                    # layer_embeddings = iter_embeddings
                    output_factors_distribution.append(A_factors)

            # sum messages of neighbors, [n_users+n_items, embed_size]
            side_embeddings = torch.cat(layer_embeddings, 1)

            ego_embeddings = side_embeddings
            # concatenate outputs of all layers
            all_embeddings += [ego_embeddings]

        all_embeddings = torch.stack(all_embeddings, 1)
        all_embeddings = torch.mean(all_embeddings, dim=1)

        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.num_users, self.num_items], 0)

        return u_g_embeddings, i_g_embeddings

    def forward(self, inputs, **kwargs):
        gu, gi = inputs
        gamma_u = torch.squeeze(gu).to(self.device)
        gamma_i = torch.squeeze(gi).to(self.device)

        xui = torch.sum(gamma_u * gamma_i, 1)

        return xui

    def predict(self, gu, gi, **kwargs):
        return torch.matmul(gu.to(self.device), torch.transpose(gi.to(self.device), 0, 1))

    @staticmethod
    def get_loss_ind(x1, x2):
        # reference: https://recbole.io/docs/_modules/recbole/model/general_recommender/dgcf.html
        def _create_centered_distance(x):
            r = torch.sum(x * x, dim=1, keepdim=True)
            v = r - 2 * torch.mm(x, x.T + r.T)
            z_v = torch.zeros_like(v)
            v = torch.where(v > 0.0, v, z_v)
            D = torch.sqrt(v + 1e-8)
            D = D - torch.mean(D, dim=0, keepdim=True) - torch.mean(D, dim=1, keepdim=True) + torch.mean(D)
            return D

        def _create_distance_covariance(d1, d2):
            v = torch.sum(d1 * d2) / (d1.shape[0] * d1.shape[0])
            z_v = torch.zeros_like(v)
            v = torch.where(v > 0.0, v, z_v)
            dcov = torch.sqrt(v + 1e-8)
            return dcov

        D1 = _create_centered_distance(x1)
        D2 = _create_centered_distance(x2)

        dcov_12 = _create_distance_covariance(D1, D2)
        dcov_11 = _create_distance_covariance(D1, D1)
        dcov_22 = _create_distance_covariance(D2, D2)

        # calculate the distance correlation
        value = dcov_11 * dcov_22
        zero_value = torch.zeros_like(value)
        value = torch.where(value > 0.0, value, zero_value)
        loss_ind = dcov_12 / (torch.sqrt(value) + 1e-10)
        return loss_ind

    def train_step(self, batch, cor_users, cor_items):
        users, pos, neg = batch

        ua_embeddings, ia_embeddings = self.propagate_embeddings(pick_=self.is_pick)

        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos]

        neg_i_g_embeddings = ia_embeddings[neg]
        u_g_embeddings_pre = self.Gu[users]
        pos_i_g_embeddings_pre = self.Gi[pos]
        neg_i_g_embeddings_pre = self.Gi[neg]

        cor_u_g_embeddings = ua_embeddings[torch.tensor(cor_users, device=self.device)]
        cor_i_g_embeddings = ia_embeddings[torch.tensor(cor_items, device=self.device)]

        xui_pos = self.forward(inputs=(u_g_embeddings, pos_i_g_embeddings))
        xui_neg = self.forward(inputs=(u_g_embeddings, neg_i_g_embeddings))

        bpr_loss = torch.mean(self.softplus(-(xui_pos - xui_neg)))

        reg_loss = self.l_w_bpr * (1 / 2) * (torch.norm(u_g_embeddings_pre) ** 2
                                             + torch.norm(pos_i_g_embeddings_pre) ** 2
                                             + torch.norm(neg_i_g_embeddings_pre) ** 2) / len(users)

        # independence loss
        loss_ind = torch.tensor(0.0, device=self.device)
        if self.intents > 1 and self.l_w_ind > 1e-9:
            sampled_embeddings = torch.cat((cor_u_g_embeddings.to(self.device), cor_i_g_embeddings.to(self.device)), dim=0)
            for intent in range(self.intents - 1):
                ui_factor_embeddings = torch.split(sampled_embeddings, self.intents, 1)
                loss_ind += self.get_loss_ind(ui_factor_embeddings[intent].to(self.device),
                                              ui_factor_embeddings[intent + 1].to(self.device))
            loss_ind /= ((self.intents + 1.0) * self.intents / 2)
            loss_ind *= self.l_w_ind

        # sum and optimize according to the overall loss
        loss = bpr_loss + reg_loss + loss_ind
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().numpy()

    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), preds.to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)
