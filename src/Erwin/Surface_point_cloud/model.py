from shared.point_reference_runtime import torch, load_official_erwin
from torch import nn
from Erwin.S_Z.config import ERWIN_CONFIG
from shared.surface_query_decoder import QueryDecoder

class SurfaceReference(nn.Module):
    def __init__(self,method):
        super().__init__();self.method=method
        if method=='erwin':
            self.core=load_official_erwin()(**{**ERWIN_CONFIG,'c_in':15})
            width=ERWIN_CONFIG['c_hidden'][0]
        else:raise ValueError(method)
        self.decoder=QueryDecoder(width)

    def forward(self,batch):
        idx,mask,rot=batch.tree
        x=self.core(batch.node_features,batch.positions,batch.surface_batch_index,
            tree_idx=idx,tree_mask=mask,tree_idx_rot=list(rot))
        if len(x)!=len(batch.positions):raise RuntimeError('Surface encoder changed original token order/count')
        out=self.decoder(x,batch)
        if out.shape!=(len(batch.query_positions),2):raise RuntimeError('Query output alignment changed')
        return out
