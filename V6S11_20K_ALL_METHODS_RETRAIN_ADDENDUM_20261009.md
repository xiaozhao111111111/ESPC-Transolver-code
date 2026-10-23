# 20k mixed-source retraining addendum (2026-10-09)

This addendum records the user's explicit request to extend the paper training
plan to 20,000 training scenes and retrain every neural method. It supplements
`PAPER_EXPERIMENT_AGENT_GUIDE.md`; it does not alter or replace historical
experiments, checkpoints, datasets, or test files.

## Scope and data

- Training source A: `datasets/canonical_response_v3/full/data/train`, 10,000
  scenes, spheres/ellipsoids/boxes, `k=2`, 128 canonical nodes, 64 probe modes.
- Training source B:
  `outputs/v6s11_quick_train_20261008_v2/train`, 10,000 scenes drawn from the
  11-object v6s11 geometry subset. Its manifest records 100-mode Foldy--Lax
  labels and solver residuals, but marks the data exploratory and
  `independent_bem_validated=false`.
- The union is exactly 20,000 training scenes. Both source datasets remain
  immutable; a new, source-bound union/cache is required. No validation or test
  scenes may enter training or train-only normalization.
- Validation: retain and report separately the existing 1,000-scene canonical
  v3 validation set and the 1,000-scene v6s11 validation set. The selection
  score is the unweighted mean of the two per-scene mean scattered-complex
  relative-L2 metrics. Also preserve each source-specific score in the run log.
- Final review: after checkpoint selection is complete for all methods, run
  `datasets/canonical_response_v3/data/id_test` (1,000) and
  `datasets/canonical_response_v3/data/shape_ood` (500, Peanut) for every
  method. These tests have been inspected in earlier work and must be reported
  as cross-run review, not as newly sealed tests. Report mixed-Peanut and
  all-Peanut subgroups separately where the manifest permits.
- The resulting comparison is exploratory. In particular, the v6s11 source
  has incomplete independent BEM coverage; successful training does not
  upgrade its scientific-release status or certify all labels.

## Methods and training protocol

Retrain the seven neural methods M0--M6, with one run per method and seed 0:

| ID | Method | Forward input / distinction |
|---|---|---|
| M0 | S(Z)-Compositional-Transolver | Scattering-response representation with compositional inter-object coupling |
| M1 | S(Z)-Transolver | Same S(Z) representation with the standard Transolver trunk |
| M2 | S(Z)-Transolver++ | Same S(Z) representation with Transolver++ |
| M3 | S(Z)-PIBNet-adapted | Same S(Z) representation with the PIBNet-adapted graph model |
| M4 | Geo-matched-Transolver | 128-channel radial occupancy, compositional trunk |
| M5 | Geo-strong-Transolver | Surface point-cloud encoding, compositional trunk |
| M6 | S(Z)-Compositional-Transolver without inter-object coupling | Same local representation and trunk as M0, coupling branch disabled |

Use 200 epochs, batch size 16, AdamW, initial learning rate `1e-4`, cosine
decay to `1e-7`, weight decay `1e-5`, gradient clipping at 1.0, and the
previously specified `lambda_q=0.1` loss wherever that loss is defined. Use
the same output target, scene-normalized metrics, and checkpoint-selection
rule across applicable methods. Recompute all training-only normalization
statistics from the 20,000-scene union; never reuse one-source statistics.
For geometry-only models, geometric channels must not expose Z or S in the
forward path; the common auxiliary loss may use the saved Z only as previously
specified.

Following the user's GPU-utilization request, the seven methods may run
concurrently on the single remote RTX 3090 under a measured resource limit.
The qualified seven-process diagnostic produced 578.06 scene predictions per
second in total versus 379.10 with four processes; summed allocated tensor
peaks were 5.32 GiB. Each independent process retains batch 16 and its own
seed/optimizer/scheduler/RNG/checkpoints. The frozen original trainer remains
unchanged; the external parallel wrapper records its source hash and CPU
configuration. M0 resumes its completed checkpoint prefix with the recorded
four loader workers; new methods use two loader workers and two CPU threads.
The scheduling change does not enlarge the per-model effective batch size.
Before full
training, pass source-hash, merged-dataset, per-method forward/backward,
checkpoint-resume, and one-epoch timing/VRAM checks. Do not reduce the epoch
count or remove a method silently if the time estimate is large; report the
measured estimate and failure evidence first. Save an independent run manifest,
history, best/last checkpoint, and status for each method under a new
`outputs/v6s11_20k_all_methods_20261009/` directory.

## Operational constraints

- The v6s11 geometry sidecar at `datasets/subset11_quick_geometry_v1` has a
  passed training-interface receipt and can supply the geometry inputs for
  that source. This receipt validates the engineering interface, not physical
  ground truth.
- The old data's prior point-cloud cache and shape metadata must be verified
  against source manifests before use. Any geometry feature not available for
  a scene is a hard preflight failure for M4/M5, not a reason to substitute
  Z-derived features.
- Remote outputs and dependencies belong on `/root/autodl-tmp`; preserve
  source datasets and existing runs. Do not create a new AutoDL instance or
  alter global Python packages. The current remote is one RTX 3090 with 24 GB
  VRAM, Python 3.12/PyTorch 2.8+cu128, and 50 GB free data disk at preflight.
- Do not include credentials in this file, commands, environment manifests,
  checkpoints, or logs.
