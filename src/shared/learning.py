"""Geometry encoder, complex field losses and per-scene metrics."""
from __future__ import annotations
from dataclasses import replace
import torch
from torch import nn
def _complex(pair: torch.Tensor) -> torch.Tensor:
    return torch.complex(pair[..., 0], pair[..., 1])

def _reduce(values: torch.Tensor, scene: torch.Tensor, count: int):
    output = torch.zeros(count, dtype=values.dtype, device=values.device)
    output.index_add_(0, scene, values)
    return output

def projected_flux_difference(error_total: torch.Tensor, graph, phi_pinv: torch.Tensor):
    error = error_total.reshape(-1, 128)
    z = torch.complex(graph.z_real, graph.z_imag).reshape(-1, 128, 64)
    coefficients = torch.einsum("mn,on->om", phi_pinv, error)
    return torch.einsum("onm,om->on", z, coefficients).reshape(-1)

def loss_and_scene_metrics(prediction: torch.Tensor, graph, phi_pinv: torch.Tensor,
                           statistics: dict, lambda_q: float,
                           output_head: str = "scattered"):
    if output_head not in ("scattered", "total"):
        raise ValueError("invalid output head")
    count = len(graph.sample_node_counts)
    scene = graph.batch_index
    incident = _complex(graph.incident)
    label = _complex(graph.labels)
    scale = float(statistics["scatter_complex_rms"] if output_head == "scattered"
                  else statistics["total_complex_rms"])
    predicted = _complex(prediction) * scale
    target = label * scale
    if output_head == "scattered":
        scatter_prediction, scatter_target = predicted, target
        total_prediction, total_target = incident + predicted, incident + target
    else:
        total_prediction, total_target = predicted, target
        scatter_prediction, scatter_target = predicted - incident, target - incident
    difference = scatter_prediction - scatter_target
    numerator = _reduce(difference.abs().square(), scene, count)
    scatter_energy = _reduce(scatter_target.abs().square(), scene, count)
    incident_energy = _reduce(incident.abs().square(), scene, count)
    scatter_loss = numerator / (scatter_energy + 1.0e-6 * incident_energy).clamp_min(1.0e-20)
    q_difference = projected_flux_difference(total_prediction - total_target, graph, phi_pinv)
    q_numerator = _reduce(q_difference.abs().square(), scene, count)
    q_target = _complex(graph.q64_gt)
    q_energy = _reduce(q_target.abs().square(), scene, count)
    q_loss = q_numerator / (q_energy + float(statistics["q_loss_stabilizer"])).clamp_min(1.0e-20)
    loss = scatter_loss.mean() + lambda_q * q_loss.mean()

    total_numerator = _reduce((total_prediction - total_target).abs().square(), scene, count)
    total_energy = _reduce(total_target.abs().square(), scene, count)
    amp_numerator = _reduce((scatter_prediction.abs() - scatter_target.abs()).square(), scene, count)
    maximum = torch.zeros(count, dtype=target.real.dtype, device=target.device)
    maximum.scatter_reduce_(0, scene, scatter_target.abs(), reduce="amax", include_self=False)
    mask = scatter_target.abs() > 0.05 * maximum[scene]
    phase_error = torch.angle(scatter_prediction * scatter_target.conj()).abs()
    phase_sum = _reduce(phase_error * mask, scene, count)
    mask_count = _reduce(mask.float(), scene, count)
    metrics = {
        "scattered_rel_l2": (numerator / scatter_energy.clamp_min(1.0e-20)).sqrt(),
        "scattered_incident_normalized": (numerator / incident_energy.clamp_min(1.0e-20)).sqrt(),
        "total_rel_l2": (total_numerator / total_energy.clamp_min(1.0e-20)).sqrt(),
        "q64_derived_rel_l2": (q_numerator / q_energy.clamp_min(1.0e-20)).sqrt(),
        "scattered_amplitude_rel_l2": (amp_numerator / scatter_energy.clamp_min(1.0e-20)).sqrt(),
        "scattered_masked_phase_mae_deg": phase_sum / mask_count.clamp_min(1.0) * (180.0 / torch.pi),
    }
    return loss, metrics

def field_loss_and_scene_metrics(prediction: torch.Tensor, graph, phi_pinv: torch.Tensor, statistics: dict, lambda_q: float, output_head: str='scattered'):
    if output_head not in ('scattered', 'total'):
        raise ValueError('invalid output head')
    count = len(graph.sample_node_counts)
    scene = graph.batch_index
    incident = _complex(graph.incident)
    label = _complex(graph.labels)
    scale = float(statistics['scatter_complex_rms'] if output_head == 'scattered' else statistics['total_complex_rms'])
    predicted = _complex(prediction) * scale
    target = label * scale
    if output_head == 'scattered':
        scatter_prediction, scatter_target = (predicted, target)
        total_prediction, total_target = (incident + predicted, incident + target)
    else:
        total_prediction, total_target = (predicted, target)
        scatter_prediction, scatter_target = (predicted - incident, target - incident)
    difference = scatter_prediction - scatter_target
    numerator = _reduce(difference.abs().square(), scene, count)
    scatter_energy = _reduce(scatter_target.abs().square(), scene, count)
    incident_energy = _reduce(incident.abs().square(), scene, count)
    scatter_loss = numerator / (scatter_energy + 1e-06 * incident_energy).clamp_min(1e-20)
    loss = scatter_loss.mean()
    total_numerator = _reduce((total_prediction - total_target).abs().square(), scene, count)
    total_energy = _reduce(total_target.abs().square(), scene, count)
    amp_numerator = _reduce((scatter_prediction.abs() - scatter_target.abs()).square(), scene, count)
    maximum = torch.zeros(count, dtype=target.real.dtype, device=target.device)
    maximum.scatter_reduce_(0, scene, scatter_target.abs(), reduce='amax', include_self=False)
    mask = scatter_target.abs() > 0.05 * maximum[scene]
    phase_error = torch.angle(scatter_prediction * scatter_target.conj()).abs()
    phase_sum = _reduce(phase_error * mask, scene, count)
    mask_count = _reduce(mask.float(), scene, count)
    metrics = {'scattered_rel_l2': (numerator / scatter_energy.clamp_min(1e-20)).sqrt(), 'scattered_incident_normalized': (numerator / incident_energy.clamp_min(1e-20)).sqrt(), 'total_rel_l2': (total_numerator / total_energy.clamp_min(1e-20)).sqrt(), 'scattered_amplitude_rel_l2': (amp_numerator / scatter_energy.clamp_min(1e-20)).sqrt(), 'scattered_masked_phase_mae_deg': phase_sum / mask_count.clamp_min(1.0) * (180.0 / torch.pi)}
    return (loss, metrics)

class GeoStrongAdapted(nn.Module):
    """Raw surface-point encoder feeding the 140-channel point backbone."""

    def __init__(self, core):
        super().__init__()
        self.local_mlp = nn.Sequential(nn.Linear(6, 64), nn.SiLU(), nn.Linear(64, 128))
        self.global_mlp = nn.Sequential(nn.Linear(6, 64), nn.SiLU(), nn.Linear(64, 128))
        self.combiner = nn.Linear(384, 128)
        self.backbone = core

    def forward(self, graph):
        if graph.geo_local is None or graph.geo_global is None:
            raise ValueError("Geo-strong requires nearest16 and 2048-point tensors")
        local = self.local_mlp(graph.geo_local).amax(dim=1)
        all_global = self.global_mlp(graph.geo_global)
        global_code = torch.cat((all_global.amax(dim=1), all_global.mean(dim=1)), dim=-1)
        global_per_node = torch.repeat_interleave(global_code, 128, dim=0)
        if len(global_per_node) != len(local):
            raise ValueError("point cloud/object node alignment failed")
        code = self.combiner(torch.cat((local, global_per_node), dim=-1))
        features = graph.node_features.clone()
        features[:, :128] = code
        return self.backbone(replace(graph, node_features=features))
