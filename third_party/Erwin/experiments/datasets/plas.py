import os
import scipy.io as scio
import numpy as np
import torch
from torch_cluster import knn_graph


def num_nodes_to_batch_idx(num_nodes):
    return torch.arange(len(num_nodes)).to(num_nodes.device).repeat_interleave(num_nodes)


class PlasDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        data_path,
        split,
        knn=16,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data_path = data_path + '/plasticity/plas_N987_T20.mat'
        self.split = split
        self.knn = knn

        self.x_mean = 19.6351
        self.x_std = 2.3652
        self.y_mean = -4.4049
        self.y_std = 14.3477

        data = scio.loadmat(self.data_path)
        input = torch.tensor(data['input'], dtype=torch.float)
        output = torch.tensor(data['output'], dtype=torch.float)

        if split == 'train':
            self.x = (input[:900][:, :101].reshape(900, 101, 1).repeat(1, 1, 31).reshape(900, -1, 1) - self.x_mean) / self.x_std
            self.y = (output[:900][:, :101, :31].reshape(900, -1, 20, 4) - self.y_mean) / self.y_std
        else:
            self.x = (input[-80:][:, :101].reshape(80, 101, 1).repeat(1, 1, 31).reshape(80, -1, 1) - self.x_mean) / self.x_std
            self.y = (output[-80:][:, :101, :31].reshape(80, -1, 20, 4) - self.y_mean) / self.y_std

        x = np.linspace(0, 1, 31)
        y = np.linspace(0, 1, 101)
        x, y = np.meshgrid(x, y)
        pos = np.c_[x.ravel(), y.ravel()]
        self.pos = torch.tensor(pos, dtype=torch.float)

        self.t = torch.arange(20)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):

        pos = self.pos
        a = self.x[idx]
        u = self.y[idx]
        t = self.t

        num_timesteps = t.size(0)
        permuted_indices = torch.randperm(num_timesteps)
        t = t[permuted_indices][[0]] # random initial time
        u = u[:, t, :] # 4 time steps

        output = {
            "node_positions": pos,
            "node_features": a,
            "target": u.squeeze(1),
            "time": t.unsqueeze(0).repeat(3131, 1),
            "num_nodes": torch.LongTensor([3131]),
        }

        return output
    
    def collate_fn(self, batch):
        batch = {k: torch.cat([d[k] for d in batch], dim=0) for k in batch[0].keys()}
        batch["batch_idx"] = num_nodes_to_batch_idx(batch["num_nodes"])
        if self.knn:
            batch["edge_index"] = knn_graph(batch["node_positions"], k=self.knn, batch=batch["batch_idx"], loop=True)
        return batch