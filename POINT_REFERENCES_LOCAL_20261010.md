# Local PTv3 and Erwin reference experiment

The user authorizes training on the local GPU. These are official-implementation
backbone comparisons, not newly invented Transformer substitutes. The input is
the same 140-channel S(Z) tensor as the mixed-20k Transolver experiment. Original
data, labels, test files, prior checkpoints and frozen trainers remain unchanged.

## Fixed protocol

- 20,000 training scenes: 10,000 canonical-v3 and 10,000 eleven-body v6s11.
- Two independently laid out validation sets, 1,000 scenes each. Select best
  checkpoint only by the equal-weight mean of their per-scene complex scattered
  relative L2 means.
- One seed (0), 200 epochs, batch 16, float32, AdamW, weight decay 1e-5,
  cosine 1e-4 to 1e-7, gradient norm clipping 1, lambda_q=0.1.
- Exact original loss and metrics functions are loaded as AST leaves from
  v3_learning.py, avoiding unused DGL/FEM dependencies. All predictions are
  restored to physical units before computing scene-level metrics.
- Train-only mixed20k input/output scales are reused, never fitted to tests.
- At completion, evaluate ID (1000), N8 (1000), Peanut (500), Torus (500),
  Peanut-N8 (250), Torus-N8 (250), reporting mean/median/p95 of all six metrics
  and mixed/all-OOD subgroups. Tests remain exploratory, previously viewed.

## Official sources and adaptations

- PTv3 commit `3229e9b7de1770c8ad17c316f8e349982de509f8`,
  https://github.com/Pointcept/PointTransformerV3 .
  Four encoder stages 32/64/128/256, two blocks per stage, decoder 64/64/128,
  patch size 128, official sparse CPE and serialized pooling/unpooling retained.
  Official non-FlashAttention path is used on Windows. Existing shuffle flags
  are configured false in both serialization and pooling for deterministic
  validation. Normalized-coordinate voxel size 0.16 is NOT a FEM mesh h. Full
  collision scan must pass; no canonical output node may be dropped.
  Backbone plus regression head: 9,632,130 parameters.
- Erwin commit `cc2d206648387cd524e347ee94bfec92d43b3ca4`,
  https://github.com/maxxxzdn/erwin .
  Four stages 32/64/128/256, two blocks per encoder/decoder stage, stride 2,
  ball size 32, rotated-tree cross-ball interaction at 45 degrees. Official
  distance-biased attention, SwiGLU, pooling and unpooling retained. The authors'
  optional MPNN embedding is disabled (mp_steps=0); missing torch_cluster fails
  closed if this changes. Backbone plus regression head: 3,686,650 parameters.
- Official balltree-erwin 0.1.0 Cython builds crash on this Windows CPython /
  LLVM-MinGW combination. The alternative binding uses a minimal C ABI to the
  same downloaded C++ header; only `long` -> `intptr_t` is required for Windows
  LLP64. The Python rotation/coarsening wrapper is executed verbatim from the
  official pyx. The original tarball, header and failed Cython builds remain.
  Each scene's tree is built separately and cached, ensuring padding, rotation,
  pooling and attention cannot mix different scenes or depend on batch layouts.
- A shared pointwise regression head produces real/imaginary scattered field
  at the original 128*N nodes. These are adapted point-regression configurations,
  not the authors' large semantic-segmentation checkpoint configurations, and
  parameter counts are not matched.

## Execution and evidence

Scripts are separate from prior training. Dependencies are isolated in
work/point_reference_deps_20261010 and use the working Torch 2.8 CUDA 12.8
package runtime. Windows compilation affects no global Python installation.

Before production: verify all used cache hashes, normalization identity, all
scene voxel uniqueness, scene isolation, no auxiliary-label/Z forward leakage,
full-batch forward/backward, finite/clipped gradients, optimizer/scheduler
restoration and N8 output alignment. Preflight weights are NOT training weights.
Production initializes anew at seed zero.

Outputs: outputs/point_references_20k_20261010. Each method has independent
run_manifest, status, history, complete last checkpoint (including RNG/loader
state), validation-selected best checkpoint and six test result files.
The supervisor confirms both child identities; missing tool handles alone never
permit duplicate launches. Simultaneous execution is allowed only after combined
resource checks; preserve batch and every frozen training setting.

This execution does not upgrade the exploratory v6s11 numerical release, prove
unrestricted geometry/frequency generalization, or supply multi-seed evidence.

## Qualified runtime memory policy

Both workers started successfully and wrote actual batch progress and epoch
checkpoints. Combined GPU usage was about 2.4 GiB, but each process's resident
working set grew to roughly 10-11 GiB as it mapped the shared read-only arrays;
system available RAM briefly fell below 1 GiB. The separate, identity-checked
`bound_point_reference_working_sets.py` applies a 4.5-GiB resident working-set
maximum to each of these two workers only. This controls paging of mapped cache
pages, not virtual tensor allocation, GPU memory, dtype, model, batches, loss or
optimizer. Available RAM immediately recovered to about 3.3 GiB. Prior limits,
target PIDs/creation times and after-state are saved in `working_set_policy.json`.
No unrelated process was closed, suspended or changed. If recovered workers
have new identities, requalify/reapply this execution-only policy rather than
changing the frozen training code. See the
[Windows API semantics](https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-setprocessworkingsetsizeex).
