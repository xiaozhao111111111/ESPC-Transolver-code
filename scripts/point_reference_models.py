"""Canonical-node regression adapters around unchanged official backbones."""
from point_reference_runtime import torch, load_official_ptv3, load_official_erwin
from torch import nn

# Explicit, four-stage configurations for the 8-GiB local GPU. Not authors'
# large semantic-segmentation checkpoints; all official operations are retained.
PT_CONFIG = dict(in_channels=140, order=('z', 'z-trans', 'hilbert', 'hilbert-trans'),
                 stride=(2, 2, 2), enc_depths=(2, 2, 2, 2), enc_channels=(32, 64, 128, 256),
                 enc_num_head=(2, 4, 8, 16), enc_patch_size=(128, 128, 128, 128),
                 dec_depths=(2, 2, 2), dec_channels=(64, 64, 128), dec_num_head=(4, 4, 8),
                 dec_patch_size=(128, 128, 128), drop_path=0.0, shuffle_orders=False,
                 enable_flash=False, enable_rpe=False)
ERWIN_CONFIG = dict(c_in=140, c_hidden=[32, 64, 128, 256], enc_num_heads=[2, 4, 8, 16],
                    enc_depths=[2, 2, 2, 2], dec_num_heads=[2, 4, 8], dec_depths=[2, 2, 2],
                    strides=[2, 2, 2], ball_sizes=[32, 32, 32, 32],
                    rotate=45, dimensionality=3, mlp_ratio=4, mp_steps=0)
GRID_SIZE = 0.16  # normalized-coordinate voxel, not a FEM mesh size


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
        elif method == 'erwin':
            self.core = load_official_erwin()(**ERWIN_CONFIG)
            width = ERWIN_CONFIG['c_hidden'][0]
        else:
            raise ValueError(method)
        self.head = nn.Sequential(nn.Linear(width, 64), nn.GELU(), nn.Linear(64, 2))

    def forward(self, batch):
        if self.method == 'ptv3':
            x = self.core(dict(feat=batch.node_features, coord=batch.positions,
                               grid_coord=batch.grid, batch=batch.batch_index)).feat
        else:
            idx, mask, rotations = batch.tree
            x = self.core(batch.node_features, batch.positions, batch.batch_index,
                          tree_idx=idx, tree_mask=mask, tree_idx_rot=list(rotations))
        output = self.head(x)
        if output.shape != batch.labels.shape:
            raise RuntimeError('Canonical-node output mapping changed')
        return output
