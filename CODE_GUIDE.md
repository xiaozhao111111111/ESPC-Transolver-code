# ESPC-Transolver source code guide

_English guide to the source-only experiment package and its reproduction prerequisites._

---

This package contains source, configuration, dependency references, and experiment
protocols. It excludes training/test caches, raw samples, trained weights, results,
and installed Python/Torch environments. Original experiment data are not modified
or distributed by this release.

## 📋 Entry points for the five methods

| Method | Experiment ID | Model or input implementation | Training entry point |
|---|---|---|---|
| ESPC-Transolver | M0 | [Compositional model](src/cspgnet/v3_compositional_transolver.py) and [object module](src/cspgnet/v3_object_transolver.py) | `scripts/train_v6s11_20k_all_methods.py --method M0` |
| Transolver-Geo-strong | M5 | `M5CompositionalGeoStrong` in the [shared trainer](scripts/train_v6s11_20k_all_methods.py) | Same trainer with `--method M5` |
| Transolver-Geo-matched | M4 | Same compositional backbone; `PreparedScenes` replaces response input with occupancy descriptors | Same trainer with `--method M4` |
| PIBNet | M3 | [PIBNet backbone](src/pibnet/gnn/pibnet.py) and [edge-index backend](src/cspgnet/official_pibnet_edge_backend.py) | Same trainer with `--method M3` |
| Geo-strong Erwin | Separate Geo-strong experiment | [Geometry adapter](scripts/point_geo_strong_erwin.py), [canonical-node model](scripts/point_reference_models.py), and [Erwin backbone](third_party/Erwin/models/erwin.py) | `scripts/train_geo_strong_erwin_local.py` |

The M0/M4/M5 compositional implementations are preserved from the existing
experiments. The earlier `TransolverGeoStrong` in
[v3_transolver_representations.py](src/cspgnet/v3_transolver_representations.py)
must not be confused with `M5CompositionalGeoStrong`.

## 📚 Supporting source locations

- Shared geometry encoder: `GeoStrongAdapted` in
  [v3_learning.py](src/cspgnet/v3_learning.py). M5 uses the corresponding
  local/global encoding structure.
- Upstream backbones: `third_party/Transolver/`, `third_party/Transolver_plus/`,
  and `third_party/Erwin/`.
- Preprocessing and evaluation:
  [prepare_v6s11_20k_all_methods.py](scripts/prepare_v6s11_20k_all_methods.py),
  [evaluate_torus_20k_review.py](scripts/evaluate_torus_20k_review.py), and
  [prepare_point_reference_tests.py](scripts/prepare_point_reference_tests.py).
- Windows Erwin C ABI: [erwin_cabi.cpp](scripts/erwin_cabi.cpp) and
  [erwin_windows_binding.py](scripts/erwin_windows_binding.py).
- Upstream ball-tree C/Cython source:
  `work/point_reference_packages_20261010/balltree_erwin-0.1.0/`.
- Required timm source functions: `work/subset11_train_deps/timm/`.

The retained `work/` contents are source, not data or installation caches.
Compiled extensions are not included.

## 💾 Historical data and checkpoint locations

The following paths are relative to the original experiment project root.
They are provided to document the expected layout, not as downloadable artifacts;
none are included in this repository.

| Original relative path | Purpose |
|---|---|
| `outputs/v6s11_20k_all_methods_20261009/prepared_retry5` | 20,000 training scenes, 2,000 validation scenes, and some test caches |
| `outputs/v6s11_20k_all_methods_20261009/torus_review_20261010/cache` | Torus test cache |
| `outputs/point_references_20k_20261010/test_sidecar` | N8 and N8-OOD test caches |
| `outputs/point_surface_20k_20261011/geometry_v3` | Surface point clouds and normals |
| `outputs/erwin_geo_strong_local_20261013/geometry` | Erwin nearest16 neighborhoods and trees |
| `outputs/v6s11_20k_all_methods_20261009/remote_results/final_20261010` | M0/M3/M4/M5 checkpoints and results |
| `outputs/erwin_geo_strong_local_20261013/runs/erwin` | Erwin checkpoints and six-test results |
| `datasets/canonical_response_v3` | Original dataset |

## ⚙️ Reproduction prerequisites

This is a source migration, not a self-contained runnable data/environment bundle.
Model, training, loss, and preprocessing source retain their original bytes.
[CODE_MANIFEST.json](CODE_MANIFEST.json) records file hashes and provenance;
release documentation is tracked separately from original experiment source.

Before running training:

1. Install the dependencies in a separate environment and configure the official
   dependency sources and Windows ball-tree dynamic library.
2. Supply the required data and source/data identity manifests.
3. Configure historical machine-specific paths in `point_reference_runtime.py`
   and `official_pibnet_train.py`, and restore the relative data layout expected
   by Erwin. Missing files cannot be replaced with placeholder tensors.
4. Select a fresh output location. Do not overwrite reference experiments or
   bypass qualification and identity checks.

For M0/M3/M4/M5, the original entry point accepts `--cache` for the prepared cache
and `--run-root` for new training outputs. Inspect its `--help` after configuring
the environment. Do not set `--run-root` to an original reference run.

Erwin qualification, source binding, and execution order are documented in
[ERWIN_GEO_STRONG_LOCAL_20261013.md](ERWIN_GEO_STRONG_LOCAL_20261013.md).
Additional recorded settings are retained in the other experiment protocol files.

[requirements_model_runtime.txt](requirements_model_runtime.txt) summarizes core
dependencies from the recorded environment.
[original_requirements/requirements.txt](original_requirements/requirements.txt)
contains the original repository requirements. Neither is a validated
cross-platform lock file.

## 🔍 Source verification

```bash
python -B inspect_sources.py
```

This checks source checksums and Python syntax without loading Torch or training.
It does not validate the data, installed environment, numerical predictions, or
reproducibility of a complete experiment. No new environment installation,
training, or model evaluation is performed as part of this documentation update.

## ⚠️ Experimental interpretation

The original M0/M3/M4/M5 experiments use auxiliary loss λq = 0.1; local Geo-strong
Erwin uses field-only loss with λq = 0. The recorded protocol uses seed 0,
200 epochs, and batch size 16. Inputs, decoders, parameter counts, and some losses
differ, so the comparison is not an equal-parameter, pure input ablation.

Labels remain exploratory where formal BEM/truncation gates are incomplete.
Migrating or translating source does not establish that all physical audits have
passed. Preserve component-specific licenses and attribution as described in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
