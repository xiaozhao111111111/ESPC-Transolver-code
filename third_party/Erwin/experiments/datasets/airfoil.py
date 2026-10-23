import os
import numpy as np
import torch
from torch_cluster import knn_graph


def num_nodes_to_batch_idx(num_nodes):
    return torch.arange(len(num_nodes)).to(num_nodes.device).repeat_interleave(num_nodes)


class AirfoilDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        data_path,
        split,
        knn=16,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.split = split
        self.knn = knn
        self.mean = 0.8163
        self.std = 0.1385

        self.pos_mean = torch.tensor([1.4925e+00, 3.4233e-04]).reshape(1,1,2)
        self.pos_std = torch.tensor([6.6607, 6.0777]).reshape(1,1,2)

        INPUT_X = data_path + '/airfoil/naca/NACA_Cylinder_X.npy'
        INPUT_Y = data_path + '/airfoil/naca/NACA_Cylinder_Y.npy'
        OUTPUT_Sigma = data_path + '/airfoil/naca/NACA_Cylinder_Q.npy'

        inputX = np.load(INPUT_X)
        inputX = torch.tensor(inputX, dtype=torch.float)
        inputY = np.load(INPUT_Y)
        inputY = torch.tensor(inputY, dtype=torch.float)
        input = torch.stack([inputX, inputY], dim=-1)

        output = np.load(OUTPUT_Sigma)[:, 4]
        output = torch.tensor(output, dtype=torch.float)

        if split == 'train':
            self.x = (input[:1000][:, :221, :51].reshape(1000, -1, 2) - self.pos_mean) / self.pos_std
            self.y = (output[:1000][:, :221, :51].reshape(1000, -1, 1) - self.mean) / self.std
        else:
            self.x = (input[1000:1000 + 200][:, :221, :51].reshape(200, -1, 2) - self.pos_mean) / self.pos_std
            self.y = (output[1000:1000 + 200][:, :221, :51].reshape(200, -1, 1)  - self.mean) / self.std

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        output = {
            "node_positions": self.x[idx],
            "target": self.y[idx],
            "num_nodes": torch.LongTensor([11271]),
        }

        return output

    def collate_fn(self, batch):
        batch = {k: torch.cat([d[k] for d in batch], dim=0) for k in batch[0].keys()}
        batch["batch_idx"] = num_nodes_to_batch_idx(batch["num_nodes"])
        if self.knn:
            batch["edge_index"] = knn_graph(batch["node_positions"], k=self.knn, batch=batch["batch_idx"], loop=True)
        return batch