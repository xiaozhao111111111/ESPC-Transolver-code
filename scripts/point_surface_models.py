"""Official surface encoders plus one shared canonical query attention decoder."""
import math
from point_reference_runtime import torch,load_official_ptv3,load_official_erwin
from point_reference_models import PT_CONFIG,ERWIN_CONFIG
from torch import nn

class QueryDecoder(nn.Module):
    def __init__(self,width):
        super().__init__();self.heads=4;self.width=64
        self.query=nn.Sequential(nn.Linear(12,64),nn.GELU(),nn.Linear(64,64))
        self.key=nn.Linear(width,64);self.value=nn.Linear(width,64)
        self.relative=nn.Sequential(nn.Linear(3,64),nn.GELU(),nn.Linear(64,64))
        self.norm=nn.LayerNorm(64)
        self.head=nn.Sequential(nn.Linear(64,64),nn.GELU(),nn.Linear(64,2))

    def forward(self,surface,batch):
        # All kNN indexes were built within each scene, never across a batch.
        idx=batch.neighbors;n,k=idx.shape
        delta=batch.positions[idx]-batch.query_positions[:,None,:]
        geo=self.relative(delta)
        q=self.query(batch.query_features)
        key=(self.key(surface)[idx]+geo).reshape(n,k,4,16)
        value=(self.value(surface)[idx]+geo).reshape(n,k,4,16)
        score=(q.reshape(n,1,4,16)*key).sum(-1)/math.sqrt(16)
        weights=score.softmax(dim=1)
        result=(weights[...,None]*value).sum(1).reshape(n,64)
        return self.head(self.norm(q+result))

class SurfaceReference(nn.Module):
    def __init__(self,method):
        super().__init__();self.method=method
        if method=='ptv3':
            cfg={**PT_CONFIG,'in_channels':15}
            self.core=load_official_ptv3()(**cfg)
            for module in self.core.modules():
                if module.__class__.__name__=='SerializedPooling':module.shuffle_orders=False
            width=PT_CONFIG['dec_channels'][0]
        elif method=='erwin':
            self.core=load_official_erwin()(**{**ERWIN_CONFIG,'c_in':15})
            width=ERWIN_CONFIG['c_hidden'][0]
        else:raise ValueError(method)
        self.decoder=QueryDecoder(width)

    def forward(self,batch):
        if self.method=='ptv3':
            x=self.core(dict(feat=batch.node_features,coord=batch.positions,
                grid_coord=batch.grid,batch=batch.surface_batch_index)).feat
        else:
            idx,mask,rot=batch.tree
            x=self.core(batch.node_features,batch.positions,batch.surface_batch_index,
                tree_idx=idx,tree_mask=mask,tree_idx_rot=list(rot))
        if len(x)!=len(batch.positions):raise RuntimeError('Surface encoder changed original token order/count')
        out=self.decoder(x,batch)
        if out.shape!=(len(batch.query_positions),2):raise RuntimeError('Query output alignment changed')
        return out
