from shared.point_reference_runtime import torch, load_official_erwin
from torch import nn
from Erwin.S_Z.config import ERWIN_CONFIG

class PointReference(nn.Module):
    def __init__(self, method):
        super().__init__()
        self.method = method
        if method == 'erwin':
            self.core = load_official_erwin()(**ERWIN_CONFIG)
            width = ERWIN_CONFIG['c_hidden'][0]
        else:
            raise ValueError(method)
        self.head = nn.Sequential(nn.Linear(width, 64), nn.GELU(), nn.Linear(64, 2))

    def forward(self, batch):
        idx, mask, rotations = batch.tree
        x = self.core(batch.node_features, batch.positions, batch.batch_index,
                      tree_idx=idx, tree_mask=mask, tree_idx_rot=list(rotations))
        output = self.head(x)
        if output.shape != batch.labels.shape:
            raise RuntimeError('Canonical-node output mapping changed')
        return output
