# User-authorized local Geo-strong Erwin

The user explicitly replaces direct-surface Erwin input with the existing
Geo-strong local/global point-cloud encoding. This is a new experiment at
`outputs/erwin_geo_strong_local_20261013`, not an input switch in old weights.
Preserve every previous direct-surface/S(Z) result, source and failure record;
do not restart PTv3. Stop the superseded direct-surface worker through its
verified managed execution session, preserving its last complete checkpoint.

Use all 2048 actual saved surface samples and normals per body, saved rotations
with the existing Geo-strong SVD correction, and 16 nearest within-body surface
points per canonical query. Reuse the exact existing GeoStrongAdapted MLPs:
6->64->128 local max, 6->64->128 global max+mean, 384->128 combination. Concatenate
the same twelve known incident/query conditions, then apply the unchanged
canonical 140-channel Erwin and its 32->64->2 output head. There are 128 main
backbone nodes per object, not one node per surface sample. The old 64-neighbor
surface-to-query decoder is absent; disclose this architectural difference.

Keep mixed20k, both source-specific 1000-scene validations, six test manifests,
field-only lambda_q=0, float32, seed0, batch16, 200 epochs, original AdamW/cosine,
clipping, train-only scales and validation-only checkpoint selection. Load no
S(Z), Z or q tensors. Reserved geometry feature slots are replaced by the
encoder's output before the backbone, not fabricated physical response tensors.

Bind all used arrays to the existing qualified field-only inventory and verify
every source SHA. Create new nearest16 and canonical-tree sidecars; never
overwrite existing source arrays or accept an incomplete cache. Qualify all
split shapes/geometry/query alignment, original-encoder equivalence, no-label
leakage, scene isolation, actual GPU batch16/N8, exact next-update checkpoint
restoration including optimizer/scheduler/all RNG, sampler order, and 200 real
seed0 training batches before production. Deterministic algorithms are used
only for controlled qualification; production retains the original setting.

Run one supervised GPU worker, retain at least 1 GiB available system RAM and
512 MiB disk. Periodically release unused CUDA cache, not live tensors, and
record that execution detail. Never fill VRAM with dummy tensors or change
batch/precision/point counts to hide a failure. Verify PID, creation time and
full command; a missing external sandbox PID or tool handle does not prove exit.
Resume only a complete source/data-matching last checkpoint with all states.

Completion requires 200 history entries, validation-only best, checkpoint SHA
and source/data provenance, and six split counts 1000/1000/500/500/250/250 with
OOD mixed/all groups. Report scattered/total complex L2 mean/median/P95, actual
whole-epoch and train-only throughput, and test time. These are single-seed
exploratory results on previously viewed, partly exploratory labels. No flux
metrics or upgraded BEM release claim. This changes encoder/decoder as well
as representation; it is not a pure input-only ablation of earlier models.
