# ESPC-Transolver-code

_Source-only release of equivalent-surface scattering experiments and geometry-based reference methods._

---

This repository collects model implementations, input adapters, training and
evaluation scripts, and preprocessing code for ESPC-Transolver,
Transolver-Geo-strong, Transolver-Geo-matched, PIBNet, and Geo-strong Erwin.

It does **not include datasets, data caches, trained weights, experiment results,
or an installed Python/CUDA environment**. Environment setup, compatible data,
and path configuration are required; this is not a one-command reproduction package.

## 📋 Methods and source code

Parameter counts refer to the configurations recorded in the existing experiments.
The methods are not parameter-matched.

| Method | ID | Main implementation | Parameters | Auxiliary loss λq |
|---|---|---|---:|---:|
| ESPC-Transolver | M0 | [Compositional backbone](src/cspgnet/v3_compositional_transolver.py), [inter-object module](src/cspgnet/v3_object_transolver.py) | 2,076,765 | 0.1 |
| Transolver-Geo-strong | M5 | [M5CompositionalGeoStrong](scripts/train_v6s11_20k_all_methods.py) | 2,143,581 | 0.1 |
| Transolver-Geo-matched | M4 | Same compositional backbone; geometry-occupancy input in [PreparedScenes](scripts/train_v6s11_20k_all_methods.py) | 2,076,765 | 0.1 |
| PIBNet | M3 | [Official backbone](src/pibnet/gnn/pibnet.py), [edge-index adapter](src/cspgnet/official_pibnet_edge_backend.py) | 5,220,930 | 0.1 |
| Geo-strong Erwin | Separate experiment | [Geometry adapter](scripts/point_geo_strong_erwin.py), [canonical-node model](scripts/point_reference_models.py), [Erwin backbone](third_party/Erwin/models/erwin.py) | 3,753,466 | 0 |

The main input differences are:

- **M0 / ESPC-Transolver:** S(Z) features constructed from the existing scattering
  response, together with known incident-field and query conditions.
- **M4 / Geo-matched:** Geometry-occupancy descriptors replace response features,
  while retaining the corresponding backbone and input-channel interface.
- **M5 / Geo-strong:** Actual surface points and normals pass through a local/global
  encoder before entering the compositional backbone.
- **M3 / PIBNet:** Response features pass through the official input and graph
  adapters, using the existing edge-index execution backend.
- **Geo-strong Erwin:** Each object's 2,048 surface points and normals are encoded
  using nearest16 neighborhoods onto 128 canonical nodes. Each node has 128
  geometry channels and 12 known-condition channels, giving 140 input channels.
  This experiment does not load S(Z), Z, or q.

The earlier, noncompositional `TransolverGeoStrong` implementation in
[v3_transolver_representations.py](src/cspgnet/v3_transolver_representations.py)
is **not the same model** as `M5CompositionalGeoStrong`. Do not interchange their
identifiers or checkpoints.

## 📦 Repository layout

```text
src/cspgnet/          Equivalent-surface models, representations, losses, and adapters
src/pibnet/           PIBNet backbone and related source
scripts/             Training, evaluation, preprocessing, and local execution scripts
third_party/         Upstream backbone source and accompanying licenses
work/                Required timm functions and ball-tree C/Cython source only
original_requirements/ Original dependency references and license notice
CODE_MANIFEST.json   File sizes, SHA-256 checksums, and relative provenance
inspect_sources.py   Standard-library-only integrity and syntax checker
CODE_GUIDE.md        Detailed English source map and reproduction prerequisites
```

The retained `work/` files are source code, not installed packages or data caches.
The original [Chinese code guide](README_代码说明.md) remains available as an optional reference.

## 🔍 Quick source check

```bash
git clone https://github.com/xiaozhao111111111/ESPC-Transolver-code.git
cd ESPC-Transolver-code
python -B inspect_sources.py
```

The checker requires no Torch installation. It verifies manifest SHA-256 checksums,
parses Python source, and checks for data, checkpoint, or runtime files that should
not be included. It does not load a model, access experiment data, or start training.

The `.gitattributes` file preserves source line endings to avoid invalidating
source-bound hashes through automatic Git newline conversion. Passing this check
does **not establish numerical accuracy, environment compatibility, or successful
training reproduction**.

## ⚙️ Environment and data configuration

1. Configure a separate environment using the
   [core dependency reference](requirements_model_runtime.txt) and the
   [original requirements](original_requirements/requirements.txt).
   The recorded model environment used Python 3.12 and Torch 2.8.0 with CUDA 12.8.
   CUDA extensions must match the target platform. Neither requirements file is
   a validated cross-platform lock file.
2. Supply compatible data caches, training statistics, and identity manifests.
   This repository does not provide a dataset download endpoint.
3. Configure the paths used by the scripts. In particular,
   `point_reference_runtime.py` and `official_pibnet_train.py` retain historical
   machine-specific dependency paths. Erwin also expects the original relative
   data layout and Windows ball-tree build artifacts.
4. Keep the original source/data identity checks. Do not bypass them with
   placeholder arrays or overwrite reference experiment outputs.

### Recorded data protocol; data not included

| Partition | Scenes |
|---|---:|
| canonical-v3 / v6s11 training | 10,000 + 10,000 |
| Two independent validation sets | 1,000 + 1,000 |
| ID / N8 tests | 1,000 / 1,000 |
| Shape OOD / Topology OOD tests | 500 / 500 |
| Shape-N8 OOD / Topology-N8 OOD tests | 250 / 250 |

See the [English code guide](CODE_GUIDE.md) for historical relative cache paths
and preparation boundaries. No dataset or checkpoint release is implied.

## 🔧 Training and evaluation entry points

After configuring the environment and data paths, M0, M3, M4, and M5 use the same
training entry point. For example:

```bash
python scripts/train_v6s11_20k_all_methods.py \
  --cache /path/to/prepared_retry5 \
  --run-root /path/to/new_runs \
  --method M0 --seed 0 --workers 0
```

Replace `M0` with `M3`, `M4`, or `M5` to select another method. Single-method mode
does not automatically evaluate all six test sets; the evaluation workflow must
be configured separately according to the original protocol.

Related preprocessing and evaluation entry points include:

- [prepare_v6s11_20k_all_methods.py](scripts/prepare_v6s11_20k_all_methods.py)
- [prepare_point_reference_tests.py](scripts/prepare_point_reference_tests.py)
- [evaluate_torus_20k_review.py](scripts/evaluate_torus_20k_review.py)

Geo-strong Erwin uses
[train_geo_strong_erwin_local.py](scripts/train_geo_strong_erwin_local.py):

```bash
python scripts/train_geo_strong_erwin_local.py --qualify
python scripts/train_geo_strong_erwin_local.py --train
```

These commands require the configured Windows/CUDA dependencies, data layout,
dynamic libraries, and source-bound identities. Qualification must pass before
production training; do not skip it or launch duplicate workers.

The recorded protocols are available in the
[Geo-strong Erwin experiment notes](ERWIN_GEO_STRONG_LOCAL_20261013.md) and the
[mixed20k retraining addendum](V6S11_20K_ALL_METHODS_RETRAIN_ADDENDUM_20261009.md).

## ⚠️ Interpretation and limitations

- The recorded protocol uses seed 0, 200 epochs, and batch size 16. Best
  checkpoints are selected using validation data only.
- M0, M3, M4, and M5 use auxiliary loss λq = 0.1. Geo-strong Erwin uses field-only
  loss with λq = 0 and has no flux metric.
- Backbones, parameter counts, encoders/decoders, and some losses differ. These
  experiments are not equal-parameter or pure input-representation ablations.
- Test results have previously been viewed. Single-seed comparisons are exploratory.
- Formal BEM and truncation requirements remain incomplete for some labels.
  This release does not establish that all publication-grade physical validation
  gates have passed.
- Packaging and translating the documentation do not rerun training or model tests.

## 🔗 Third-party attribution and licensing

Preserve upstream attribution and the license files associated with each component.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

This release does not apply a new blanket license to all code or extend the
permissions granted by third-party authors. Upstream source and historical
experiment records are preserved, including their original-language comments.
