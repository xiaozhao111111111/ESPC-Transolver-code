import torch
import torch.nn as nn


class ElasModel(nn.Module):
    def __init__(self, main_model):
        super().__init__()
        self.loss_reduce = "sum"
        self.pos_enc = nn.Linear(main_model.dimensionality, main_model.in_dim)
        self.main_model = main_model
        self.pred_head = nn.Sequential(
            nn.Linear(main_model.out_dim, main_model.out_dim),
            nn.GELU(),
            nn.Linear(main_model.out_dim, 1),
        )

        self.y_mean_denorm = 187.7388
        self.y_std_denorm = 127.0829

    def forward(self, node_positions, **kwargs):
        node_features = self.pos_enc(node_positions)
        return self.pred_head(self.main_model(node_features, node_positions, **kwargs))

    def relative_l2(self, pred, target, batch_idx):
        assert pred.shape == target.shape, f"pred: {pred.shape}, target: {target.shape}"

        batch_size = batch_idx.max().item() + 1
        pred = pred.reshape(batch_size, -1) * self.y_std_denorm + self.y_mean_denorm
        target = target.reshape(batch_size, -1) * self.y_std_denorm + self.y_mean_denorm

        diff_norms = torch.norm(pred - target, p=2, dim=1)
        y_norms = torch.norm(target, p=2, dim=1)

        return torch.sum(diff_norms / y_norms)

    def step(self, batch, prefix="train"):
        pred = self(**batch).squeeze(-1)
        loss = self.relative_l2(pred, batch["target"], batch["batch_idx"])
        return {f"{prefix}/loss": loss, f"{prefix}/rmse": loss}

    def training_step(self, batch):
        return self.step(batch, "train")

    @torch.no_grad()
    def validation_step(self, batch):
        return self.step(batch, "val")
