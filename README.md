# ESPC-Transolver

Scattering-field prediction with ESPC-Transolver and seven comparison methods.
Implementations are grouped by method; each has one public training script.

## Layout

```text
scripts/          Eight method entry points
src/              Models, input adapters and shared training utilities
data/             Prepared arrays, normalization and split definitions
runtime/          Optional platform-specific local dependencies
outputs/          Created by your own runs
requirements.txt  Python dependency reference
```

The source package contains no historical training logs, pretrained weights,
experiment receipts or machine-specific project paths.

## Environment

Use Python 3.12 with CUDA-enabled PyTorch. The optional local `runtime/` folder
contains Windows x64 / Python 3.12 / Torch 2.8 CUDA dependencies. A Python
interpreter and compatible NVIDIA driver are still required. Do not use these
binaries on another platform or Python version.

Without the bundled runtime, create and activate a virtual environment, then run:

```sh
python -m pip install -r requirements.txt
```

PTv3 also needs compatible `spconv` and `torch-scatter` binaries. Erwin's native
ball-tree binding is described in `src/Erwin/shared/README.md`. Its configured
message-passing steps are zero, so `torch-cluster` is not required.
FlashAttention is not required by the included configurations.
Compiled extensions are platform-dependent: the requirements file is not a
universal binary lock. The full local package is the ready-to-use option on its
matching platform.

## Data

Place the prepared data package in `data/`, or select it with `--data-dir`.

| Folder | Contents |
|---|---|
| `data/canonical/` | Canonical features, fields, auxiliary arrays and geometry |
| `data/topology/` | Topology-shift tests |
| `data/extended/` | Larger-object-count tests |
| `data/surface/` | Surface points, normals, query conditions and mappings |
| `data/geo_strong/` | Nearest-neighbor and canonical tree mappings |
| `data/test_layouts/` | Test ordering and OOD group membership |
| `data/manifest.json` | Relative paths, sizes and SHA-256 checksums |

`train_v3` and `train_v6s11` identify the two training sources; `val_v3` and
`val_v6s11` are their independent validation sets. The manifest lists all split
sizes. Retain the supplied train-only normalization statistics.

The full local folder includes these arrays. Git excludes data and runtime
binaries: a source-only clone is **not** the complete runnable package.
Distribute `data/` alongside the source with its layout intact. No public
dataset download URL is configured in this repository.

```sh
python scripts/train_espc_transolver.py --check-data
```

This verifies SHA-256 checksums. Add `--sizes-only` for a quick presence/size
check. `--check-sources` parses all Python files; it does not execute training.

## Methods

| Method | Script in `scripts/` | Implementation | Input |
|---|---|---|---|
| ESPC-Transolver | `train_espc_transolver.py` | `src/ESPC_Transolver/` | S(Z) |
| Transolver | `train_transolver.py` | `src/Transolver/` | S(Z) |
| Transolver++ | `train_transolver_plus.py` | `src/Transolver_plus/` | S(Z) |
| PIBNet | `train_pibnet.py` | `src/PIBNet/` | S(Z) |
| Transolver-Geo-matched | `train_transolver_geo_matched.py` | `src/Transolver_GEO_matched/` | Occupancy |
| Transolver-Geo-strong | `train_transolver_geo_strong.py` | `src/Transolver_GEO_strong/` | Local/global surface encoding |
| PTv3 | `train_ptv3.py` | `src/PTv3/` | `geo-strong`, `surface`, `s-z` |
| Erwin | `train_erwin.py` | `src/Erwin/` | `geo-strong`, `surface`, `s-z` |

PTv3 and Erwin default to `geo-strong`. Their `S_Z/`, `Geo_strong/` and
`Surface_point_cloud/` folders contain the input branches. Geo-strong encodes
all surface points and normals into canonical nodes. Direct-surface branches
encode shared surface representatives and decode at canonical query points.

## Run

Run from the package root, substituting the desired method script:

```sh
python scripts/train_espc_transolver.py --smoke
python scripts/train_espc_transolver.py
python scripts/train_ptv3.py --input geo-strong
python scripts/train_erwin.py --input geo-strong
```

`--smoke` checks a real forward/loss/backward pass on each training source
without saving a run. Run GPU methods sequentially unless resources permit
otherwise. Use `--help` for settings and paths. A bounded functional test is:

```sh
python scripts/train_erwin.py --epochs 1 --output outputs/check
```

Your checkpoints and metrics go to `outputs/<method>/<input>/`. Use `--resume`
only with your matching output directory; existing checkpoints are not silently
overwritten. Checkpoints retain optimizer, scheduler and random-generator
states. Best-model selection uses only the two validation sets.

PTv3/Erwin evaluate all six prepared test splits after training. The other six
entry points retain their training workflow and do not automatically add a new
six-split evaluation protocol.

## Scope and licenses

Surface/Geo-strong PTv3 and Erwin use field-only loss; physical canonical
branches retain their auxiliary flux loss. These are not equal-parameter or
input-only ablations. Some labels remain exploratory; packaging does not
establish independent BEM/truncation validation.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution and licenses.
