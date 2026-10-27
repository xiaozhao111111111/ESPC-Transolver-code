from shared.point_reference_runtime import torch, load_official_ptv3
from torch import nn
from PTv3.S_Z.config import PT_CONFIG, GRID_SIZE

class PointReference(nn.Module):
    def __init__(self, method):
        super().__init__()
        self.method = method
        if method == 'ptv3':
            self.core = load_official_ptv3()(**PT_CONFIG)
            # The standalone constructor doesn't forward shuffle_orders to its
            # pooling modules. Configure their existing public flag as well so
            # validation is deterministic; no author layer statement changes.
            for module in self.core.modules():
                if module.__class__.__name__ == 'SerializedPooling':
                    module.shuffle_orders = False
            width = PT_CONFIG['dec_channels'][0]
        else:
            raise ValueError(method)
        self.head = nn.Sequential(nn.Linear(width, 64), nn.GELU(), nn.Linear(64, 2))

    def forward(self, batch):
        x = self.core(dict(feat=batch.node_features, coord=batch.positions,
                           grid_coord=batch.grid, batch=batch.batch_index)).feat
        output = self.head(x)
        if output.shape != batch.labels.shape:
            raise RuntimeError('Canonical-node output mapping changed')
        return output
