import math
from shared.point_reference_runtime import torch
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
