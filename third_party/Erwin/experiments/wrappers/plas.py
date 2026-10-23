import math
import torch
import torch.nn as nn

from functools import partial


def timestep_embedding(timesteps, dim, max_period=1000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding.squeeze(1)


class PlasModel(nn.Module):
    def __init__(self, main_model):
        super().__init__()
        self.loss_reduce = "sum"
        self.enc = nn.Sequential(nn.Linear(main_model.dimensionality + 1 + main_model.in_dim, main_model.in_dim))
        self.time_emb = partial(timestep_embedding, dim=main_model.in_dim)
        self.main_model = main_model
        self.pred_head = nn.Sequential(
            nn.Linear(main_model.out_dim, main_model.out_dim),
            nn.GELU(),
            nn.Linear(main_model.out_dim, 4),
        )

        self.y_mean_denorm = -4.4049
        self.y_std_denorm = 14.3477

    def forward(self, node_features, node_positions, time, **kwargs):
        node_features = self.enc(torch.cat([node_features, node_positions, self.time_emb(time)], -1))
        return self.pred_head(self.main_model(node_features, node_positions, **kwargs))

    def relative_l2(self, pred, target, batch_idx):
        assert pred.shape == target.shape, f"pred: {pred.shape}, target: {target.shape}"

        batch_size = batch_idx[-1].item() + 1
        pred = pred.reshape(batch_size, -1) * self.y_std_denorm + self.y_mean_denorm
        target = target.reshape(batch_size, -1) * self.y_std_denorm + self.y_mean_denorm

        diff_norms = torch.norm(pred - target, p=2, dim=1)
        y_norms = torch.norm(target, p=2, dim=1)

        return torch.sum(diff_norms / y_norms)

    def step(self, batch, prefix="train"):
        pred = self(**batch)
        loss = self.relative_l2(pred, batch["target"], batch["batch_idx"])
        return {f"{prefix}/loss": loss, f"{prefix}/rmse": loss}

    def training_step(self, batch):
        return self.step(batch, "train")

    @torch.no_grad()
    def validation_step(self, batch):
        return self.step(batch, "val")
