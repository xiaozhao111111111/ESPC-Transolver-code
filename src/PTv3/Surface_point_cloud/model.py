from shared.point_reference_runtime import torch, load_official_ptv3
from torch import nn
from PTv3.S_Z.config import PT_CONFIG, GRID_SIZE
from shared.surface_query_decoder import QueryDecoder

class SurfaceReference(nn.Module):
    def __init__(self,method):
        super().__init__();self.method=method
        if method=='ptv3':
            cfg={**PT_CONFIG,'in_channels':15}
            self.core=load_official_ptv3()(**cfg)
            for module in self.core.modules():
                if module.__class__.__name__=='SerializedPooling':module.shuffle_orders=False
            width=PT_CONFIG['dec_channels'][0]
        else:
            raise ValueError(method)
        self.decoder=QueryDecoder(width)

    def forward(self,batch):
        x=self.core(dict(feat=batch.node_features,coord=batch.positions,
            grid_coord=batch.grid,batch=batch.surface_batch_index)).feat
        if len(x)!=len(batch.positions):raise RuntimeError('Surface encoder changed original token order/count')
        out=self.decoder(x,batch)
        if out.shape!=(len(batch.query_positions),2):raise RuntimeError('Query output alignment changed')
        return out
