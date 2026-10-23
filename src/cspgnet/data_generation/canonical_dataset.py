"""PyTorch reader for the canonical-surface dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class CanonicalSurfaceDataset(Dataset):
    """Load generated CSP-GNet samples as fixed-shape tensors.

    The returned fields follow the method proposal:

    ``physics``
        Real/imaginary parts of ``Z=S Phi``, shape ``[N, N_sigma, 2*r]``.
    ``incident``
        Real/imaginary incident field on each canonical surface, shape
        ``[N, N_sigma, 2]``.
    ``centers`` / ``k``
        Object positions and wave number for the object graph.
    ``target``
        Real/imaginary total field on each canonical surface, shape
        ``[N, N_sigma, 2]``.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.files = sorted(self.root.glob("sample_*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No sample_*.npz files found in {self.root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        data = np.load(self.files[index])
        z = np.concatenate([data["Z_real"], data["Z_imag"]], axis=-1)
        incident = np.stack(
            [data["incident_surface"].real, data["incident_surface"].imag], axis=-1
        )
        target = np.stack(
            [data["u_surface_gt_real"], data["u_surface_gt_imag"]], axis=-1
        )
        return {
            "physics": torch.from_numpy(z.astype(np.float32)),
            "incident": torch.from_numpy(incident.astype(np.float32)),
            "centers": torch.from_numpy(data["centers"].astype(np.float32)),
            "k": torch.as_tensor(float(data["k"]), dtype=torch.float32),
            "incident_direction": torch.from_numpy(data["incident_direction"].astype(np.float32)),
            "interface_points": torch.from_numpy(data["interface_points"].astype(np.float32)),
            "target": torch.from_numpy(target.astype(np.float32)),
        }
