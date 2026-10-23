"""Lightweight training interface for the exploratory eleven-body data.

Uses the unchanged official Transolver computation and the original timm
initialization function. Point-set training does not construct unused graphs.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, fields
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

PROJECT = Path(__file__).resolve().parents[2]
DEPS = PROJECT / "work/subset11_train_deps"
OFFICIAL_ROOT = PROJECT / "third_party/Transolver/PDE-Solving-StandardBenchmark"
OFFICIAL_FILE = OFFICIAL_ROOT / "model/Transolver_Irregular_Mesh.py"


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_pack(root):
    root = Path(root)
    m = read(root / "trainable_manifest.json")
    if m.get("schema") != "subset11-transolver-trainable-v1" or m.get("status") != "complete":
        raise RuntimeError("Expected a complete, exploratory subset11 training pack")
    if m.get("scientific_release") is not False:
        raise RuntimeError("Exploratory pack cannot become a scientific release")
    for name, item in m["files"].items():
        if sha(root / name) != item["sha256"]:
            raise RuntimeError(f"Training file checksum differs: {name}")
    return m


def timm_initializer():
    """Load timm's pinned leaf directly, avoiding its unused vision imports."""
    path = DEPS / "timm/layers/weight_init.py"
    if not path.exists():
        raise FileNotFoundError("Install timm==0.9.2 and einops==0.8.1 in work/subset11_train_deps")
    spec = importlib.util.spec_from_file_location("_subset11_timm_weight_init", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.trunc_normal_


def load_official_model():
    """Only the dependency import is redirected; every model statement is kept."""
    sys.path[:0] = [str(DEPS), str(OFFICIAL_ROOT)]
    import einops
    if einops.__version__ != "0.8.1" or not Path(einops.__file__).resolve().is_relative_to(DEPS.resolve()):
        raise RuntimeError("Expected the pinned project-local einops 0.8.1")
    tree = ast.parse(OFFICIAL_FILE.read_text(encoding="utf-8"), filename=str(OFFICIAL_FILE))
    imports = [n for n in tree.body if isinstance(n, ast.ImportFrom) and n.module == "timm.models.layers"]
    if len(imports) != 1 or [n.name for n in imports[0].names] != ["trunc_normal_"]:
        raise RuntimeError("Official dependency import changed; review the adapter")
    tree.body.remove(imports[0])
    module = types.ModuleType("_subset11_official_transolver")
    module.__file__ = str(OFFICIAL_FILE)
    module.trunc_normal_ = timm_initializer()
    sys.modules[module.__name__] = module
    exec(compile(tree, str(OFFICIAL_FILE), "exec"), module.__dict__)
    return module.Model


@dataclass
class SceneBatch:
    node_features: torch.Tensor
    labels: torch.Tensor
    incident: torch.Tensor
    q64_gt: torch.Tensor
    z_real: torch.Tensor
    z_imag: torch.Tensor
    batch_index: torch.Tensor
    sample_node_counts: torch.Tensor
    sample_index: torch.Tensor

    def to(self, device):
        # Keep sizes on CPU so grouping does not synchronize the GPU.
        return SceneBatch(**{f.name: getattr(self,f.name) if f.name == "sample_node_counts"
                             else getattr(self,f.name).to(device,non_blocking=True) for f in fields(self)})

    def pin_memory(self):
        return SceneBatch(**{f.name:getattr(self,f.name).pin_memory() for f in fields(self)})


class Subset11Dataset(Dataset):
    def __init__(self, root, split, representation="s-response"):
        if split not in ("train", "val") or representation not in ("raw-z", "s-response"):
            raise ValueError("Supported splits are train/val; inputs are raw-z/s-response")
        self.root, self.split, self.representation = Path(root), split, representation
        self.manifest = read(self.root/"trainable_manifest.json")
        if self.manifest.get("schema") != "subset11-transolver-trainable-v1" or self.manifest.get("status") != "complete":
            raise RuntimeError("Unsupported or incomplete subset11 training version")
        self.statistics = read(self.root/"statistics.json")
        if (sha(self.root/"statistics.json") != self.manifest["statistics_sha256"] or
            self.statistics.get("fit_split") != "train" or
            self.statistics["source_train_manifest_sha256"] != self.manifest["source_binding"]["source_train_manifest_sha256"]):
            raise RuntimeError("Statistics are not bound to this training split")
        self.offsets = np.load(self.root/split/"offsets.npy",allow_pickle=False)
        if len(self.offsets) != self.manifest["splits"][split]["scenes"]+1 or np.any(np.diff(self.offsets)<2):
            raise RuntimeError("Packed object offsets are invalid")
        with np.load(self.root/"shared.npz",allow_pickle=False) as z:
            self.points = z["interface_points"].astype(np.float32)
            self.phi_pinv = torch.from_numpy(z["phi_pinv"].copy())
            self.k = float(z["k"])
        self.scales = np.repeat(np.maximum(np.asarray(self.statistics[
            "z_degree_complex_rms" if representation == "raw-z" else "s_degree_complex_rms"]),
            self.statistics["complex_scale_floor"]),np.arange(1,16,2))
        self._arrays = None

    def __len__(self):
        return len(self.offsets)-1

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_arrays"] = None
        return state

    def __getitem__(self, index):
        if self._arrays is None:
            keys = ("Z", "S", "u_total_gt", "u_inc", "q64_gt", "centers", "incident_direction")
            self._arrays = {k:np.load(self.root/self.split/f"{k}.npy",mmap_mode="r",allow_pickle=False) for k in keys}
        lo, hi = int(self.offsets[index]), int(self.offsets[index+1])
        n = hi-lo
        get = lambda key: np.asarray(self._arrays[key][lo:hi])
        response = get("Z" if self.representation == "raw-z" else "S").astype(np.complex128)/self.scales[None,None,:]
        incident = get("u_inc")
        pair = lambda a: torch.from_numpy(np.stack((a.real,a.imag),-1).astype(np.float32).reshape(-1,2))
        first = torch.from_numpy(np.concatenate((response.real,response.imag),-1).astype(np.float32).reshape(-1,128))
        inc = pair(incident)
        local = torch.from_numpy(self.points.copy()).unsqueeze(0).expand(n,-1,-1)/1.5
        centers = torch.from_numpy(get("centers").copy())
        relative = (centers-centers.mean(0,keepdim=True))/1.5
        relative = relative[:,None,:].expand(n,128,3)
        direction = torch.from_numpy(np.asarray(self._arrays["incident_direction"][index]).copy()).expand(n,128,3)
        wavelength = torch.tensor(2*torch.pi/self.k,dtype=torch.float32)
        encoding = (1/wavelength-1/6)/(1/0.6-1/6)*2-1
        features = torch.cat((first,inc,local.reshape(-1,3),relative.reshape(-1,3),
                              direction.reshape(-1,3),encoding.expand(n*128,1)),dim=-1)
        target = pair(get("u_total_gt")-incident)/float(self.statistics["scatter_complex_rms"])
        z = get("Z")
        nodes = n*128
        return SceneBatch(features,target,inc,pair(get("q64_gt")),
                          torch.from_numpy(z.real.copy().reshape(-1,64)),
                          torch.from_numpy(z.imag.copy().reshape(-1,64)),
                          torch.zeros(nodes,dtype=torch.int64),torch.tensor([nodes]),
                          torch.full((nodes,),index,dtype=torch.int64))


def collate_scenes(scenes):
    result = {f.name:torch.cat([getattr(s,f.name) for s in scenes]) for f in fields(SceneBatch)}
    result["batch_index"] = torch.repeat_interleave(torch.arange(len(scenes)),result["sample_node_counts"])
    return SceneBatch(**result)


class Subset11Transolver(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = load_official_model()(space_dim=140,fun_dim=0,out_dim=2,n_layers=5,
                                            n_hidden=256,n_head=8,slice_num=32,mlp_ratio=1,
                                            dropout=0.0,unified_pos=False)

    def forward(self, batch):
        counts = batch.sample_node_counts.tolist()
        if any(c<=0 or c%128 for c in counts):
            raise ValueError("Scenes must contain complete canonical spheres")
        scenes = torch.split(batch.node_features,counts,dim=0)
        result = [None]*len(scenes)
        for count in sorted(set(counts)):
            indices = [i for i,c in enumerate(counts) if c == count]
            predictions = self.backbone(torch.stack([scenes[i] for i in indices]),None)
            for index,prediction in zip(indices,predictions.unbind(0)):
                result[index] = prediction
        return torch.cat(result)


def loss_and_metrics(prediction, batch, phi_pinv, statistics, lambda_q=0.1):
    """Original scene-normalized scattered and derived-flux loss, in physical units."""
    n_scenes = len(batch.sample_node_counts)
    scene = batch.batch_index
    complex_pair = lambda a:torch.complex(a[...,0],a[...,1])
    def reduce(a):
        return torch.zeros(n_scenes,dtype=a.dtype,device=a.device).index_add_(0,scene,a)
    scale = float(statistics["scatter_complex_rms"])
    pred, target = complex_pair(prediction)*scale, complex_pair(batch.labels)*scale
    incident = complex_pair(batch.incident)
    difference = pred-target
    numerator = reduce(difference.abs().square())
    energy = reduce(target.abs().square())
    incident_energy = reduce(incident.abs().square())
    loss_s = numerator/(energy+1e-6*incident_energy).clamp_min(1e-20)
    z = torch.complex(batch.z_real,batch.z_imag).reshape(-1,128,64)
    # Preserve the original total-field subtraction order for numerical parity.
    total_pred, total_target = incident+pred, incident+target
    coefficients = torch.einsum("mn,on->om",phi_pinv,(total_pred-total_target).reshape(-1,128))
    q_difference = torch.einsum("onm,om->on",z,coefficients).reshape(-1)
    q_numerator = reduce(q_difference.abs().square())
    q_energy = reduce(complex_pair(batch.q64_gt).abs().square())
    loss_q = q_numerator/(q_energy+float(statistics["q_loss_stabilizer"])).clamp_min(1e-20)
    total_numerator = reduce((total_pred-total_target).abs().square())
    total_energy = reduce(total_target.abs().square())
    amplitude = reduce((pred.abs()-target.abs()).square())
    maximum = torch.zeros(n_scenes,dtype=target.real.dtype,device=target.device)
    maximum.scatter_reduce_(0,scene,target.abs(),reduce="amax",include_self=False)
    mask = target.abs()>0.05*maximum[scene]
    phase = reduce(torch.angle(pred*target.conj()).abs()*mask)/reduce(mask.float()).clamp_min(1)*180/torch.pi
    metrics = {"scattered_rel_l2":(numerator/energy.clamp_min(1e-20)).sqrt(),
               "scattered_incident_normalized":(numerator/incident_energy.clamp_min(1e-20)).sqrt(),
               "total_rel_l2":(total_numerator/total_energy.clamp_min(1e-20)).sqrt(),
               "q64_derived_rel_l2":(q_numerator/q_energy.clamp_min(1e-20)).sqrt(),
               "scattered_amplitude_rel_l2":(amplitude/energy.clamp_min(1e-20)).sqrt(),
               "scattered_masked_phase_mae_deg":phase}
    return loss_s.mean()+lambda_q*loss_q.mean(),metrics
