import torch
from ESPC_Transolver.S_Z.model import CompositionalTransolver

NODES = 128

class GeoStrongTransolver(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = CompositionalTransolver()
        self.local_mlp = torch.nn.Sequential(torch.nn.Linear(6, 64), torch.nn.SiLU(), torch.nn.Linear(64, 128))
        self.global_mlp = torch.nn.Sequential(torch.nn.Linear(6, 64), torch.nn.SiLU(), torch.nn.Linear(64, 128))
        self.combiner = torch.nn.Linear(384, 128)

    def forward(self, batch):
        if batch.geo_local is None or batch.geo_global is None:
            raise ValueError("missing transolver-geo-strong point-cloud inputs")
        local = self.local_mlp(batch.geo_local).amax(dim=1)
        enc = self.global_mlp(batch.geo_global)
        global_code = torch.cat((enc.amax(dim=1), enc.mean(dim=1)), dim=-1)
        repeated = torch.repeat_interleave(global_code, NODES, dim=0)
        if repeated.shape[0] != local.shape[0]:
            raise ValueError("transolver-geo-strong geometry/node alignment failed")
        features = batch.node_features.clone()
        features[:, :128] = self.combiner(torch.cat((local, repeated), dim=-1))
        old = batch.node_features
        batch.node_features = features
        try:
            return self.backbone(batch)
        finally:
            batch.node_features = old
