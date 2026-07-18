# 3DGS-MCMC integration

This branch contains an opt-in port of
[3D Gaussian Splatting as Markov Chain Monte Carlo](https://arxiv.org/abs/2404.09591)
onto this project's current Graphdeco/Improved-GS-compatible codebase. It is an
independent comparison with Improved-GS, not a hybrid method. The default
density controller remains unchanged:

```text
--density_control 3dgs
```

Use `--density_control mcmc` to select MCMC. MCMC and Improved-GS are separate
density-control paths: an MCMC run does not enable LAS, RAP, Growth Control,
AbsGrad, EAS, or MU.

## What the MCMC path changes

The MCMC path replaces clone/split/prune/opacity-reset densification with the
coupled procedure from the paper:

- low-opacity Gaussians are relocated to donors sampled according to activated
  opacity;
- donor and relocated opacity/scale are adjusted analytically;
- the population grows by at most `mcmc_growth_rate` per structural event until
  it reaches `cap_max`;
- SGLD position noise is applied after the dense Adam update; and
- mean opacity and scale regularizers are added to the image loss.

The rasterization path, fused SSIM, camera/exposure support, and other modern
base features are shared with the rest of this fork. MCMC adds a relocation
helper to the existing project rasterizer instead of replacing the
Improved-GS/Pixel-GS-compatible ABI. Results should therefore be described as a
controller port on a common base, not as a bit-for-bit reproduction of the
older upstream repository.

## Fresh-clone setup

The CUDA rasterizer is an upstream submodule, so the parent-owned rasterizer
patch must be applied before building the extension:

```bash
git submodule update --init --recursive \
  submodules/diff-gaussian-rasterization \
  submodules/simple-knn \
  submodules/fused-ssim
python scripts/apply_improved_gs_rasterizer_patch.py
python -m pip install --no-build-isolation --no-cache-dir --force-reinstall \
  submodules/diff-gaussian-rasterization
python -m pip install \
  submodules/simple-knn submodules/fused-ssim plyfile pycolmap lpips
```

Rebuild after every CUDA/C++ patch change. In a notebook, run the validation in
a fresh subprocess so a previously imported extension cannot mask a stale
binary.

Run the static/CPU suite and rasterizer checks before training:

```bash
python -m unittest discover -s tests -p "test_mcmc*.py" -v
python scripts/apply_improved_gs_rasterizer_patch.py --check-only
python scripts/smoke_test_mcmc_rasterizer.py
```

## Canonical HCM0204 configuration

The first experiment uses the actual 5.1M Gaussian count reached by the current
Improved-GS model, not its nominal 6M budget:

```bash
python train_scene.py \
  --input_dir /kaggle/working/cleaned_mcmc_hcm0204 \
  --model_dir /kaggle/working/mcmc_models \
  --scene_name HCM0204 \
  --iterations 30000 \
  --resolution 1 \
  --lambda_dssim 0.2 \
  --test_iterations -1 \
  --save_iterations 30000 \
  --checkpoint_iterations 9000 15000 22500 30000 \
  --density_control mcmc \
  --cap_max 5100000 \
  --densify_from_iter 500 \
  --densify_until_iter 25000 \
  --densification_interval 100 \
  --mcmc_init_type random \
  --mcmc_random_points 100000 \
  --mcmc_init_mode paper \
  --mcmc_noise_lr 500000 \
  --mcmc_opacity_reg 0.01 \
  --mcmc_scale_reg 0.01 \
  --mcmc_growth_rate 1.05 \
  --mcmc_min_opacity 0.005 \
  --mcmc_noise_chunk_size 250000 \
  --opacity_lr 0.05 \
  --optimizer_type default \
  --checkpoint_keep_last 1 \
  --stats_path /kaggle/working/mcmc_models/HCM0204/mcmc_stats.jsonl \
  --seed 0
```

There are two intentionally separate initialization controls:

- `mcmc_init_type=random` creates the upstream-default 100,000 points in the
  camera-derived cube `[-3 * radius, 3 * radius]`. `sfm` is an explicit
  ablation.
- `mcmc_init_mode=paper` initializes activated opacity to `0.5` and scale to
  `0.1 * sqrt(dist2)`. `legacy` uses this fork's normal 3DGS parameter
  initialization and is also only an ablation.

The upstream MCMC opacity learning rate is `0.05`; this command passes it
explicitly because the shared base keeps its existing default for every other
method. Resume rejects a change in any of these values.

The relocation/noise/regularization components are coupled. In particular,
the regularizers use means rather than sums, and the SGLD opacity gate follows
the implementation convention:

```python
torch.sigmoid(100.0 * (0.005 - opacity))
```

## Staged Kaggle run

[`notebooks/ver003.ipynb`](../notebooks/ver003.ipynb) is the canonical HCM0204
Kaggle experiment. It uses the following stages:

1. A 700-iteration smoke run in a separate model directory with a 400K cap.
2. The real 5.1M run stops at iteration 9000 for the resource gate. Starting
   from random 100K points, 5% growth reaches the cap on structural event 81,
   at about iteration 8600; 7000 is therefore too early for this initializer.
3. If the gate passes, resume the same run in separate subprocesses to 15000,
   22500, and 30000. The resume manifest is refreshed atomically after every
   completed stage.

The notebook pins `EXPECTED_COMMIT` to the reviewed core implementation SHA. It
checks out that commit in detached mode and fails before training if the commit
cannot be resolved exactly. The notebook itself can live in the later branch
commit because Kaggle executes the uploaded notebook while the cloned training
code remains pinned.

The resource gate keeps `--iterations 30000` and uses
`--stop_after_iteration 9000`. This preserves all 30K learning-rate schedules
and performs the normal optimizer/SGLD update at iteration 9000. Do not emulate
the gate by training with `--iterations 9000` and then changing it to 30000.

The notebook sets:

```text
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

The position-noise calculation is chunked to avoid materializing all temporary
`N x 3 x 3` tensors at once. If 5.1M still runs out of memory, first reduce the
noise chunk and inspect structural concatenation peaks. A 4.5M rerun is a new
budget ablation and must not overwrite the 5.1M experiment metadata.

## Checkpoints and exact resume

MCMC checkpoints include the Gaussian/optimizer state plus:

- Python, NumPy, Torch CPU, and Torch CUDA RNG states;
- remaining camera stack and canonical camera order;
- exposure state when applicable;
- the complete MCMC/dataset configuration, seed, and content hash of every
  selected image, depth input, and COLMAP sparse-model file;
- Python, NumPy, Torch, CUDA, GPU-device, and compute-capability version
  metadata; and
- the current absolute iteration.

Resume rejects a checkpoint from another density controller, another dataset
content fingerprint, another recorded runtime environment, or an incompatible
MCMC configuration. The notebook adds an external `resume_manifest.json` with
the scene, branch/commit, configuration fingerprint, checkpoint filename,
iteration, byte count, and SHA-256. An attached Kaggle resume bundle is verified
and copied through a temporary file before an atomic rename into the dedicated
working model directory.

Checkpoint writes and manifests are atomic and `checkpoint_keep_last=1` retains
only the newest valid checkpoint. At 5.1M Gaussians, one dense-Adam checkpoint
is roughly 3.7 GB, while the final binary PLY is roughly 1.3 GB. Keeping four
full checkpoints can consume about 15 GB before accounting for preprocessed
images, the CUDA build, and render outputs. The final non-resumable checkpoint
does not duplicate pending dense gradients.

## Experiment order

Use the same cleaned HCM0204 train/test split, 30K schedule, 5.1M cap, seed,
renderer, and Q96 4:4:4 post-processing contract as Improved-GS. Change one
factor at a time:

| Priority | Run | Purpose |
|---:|---|---|
| 0 | Smoke 700, cap 400K | Verify CUDA relocation, finite loss, growth, stats, checkpoint, and resume. |
| 1 | Random100K + paper init + cap 5.1M, seed 0 | Canonical MCMC result and direct comparison with Improved-GS. |
| 2 | Same checkpoint, bicubic sharpen 0 and 0.3 at Q96 | Separate model quality from the already successful post-process gain. |
| 3 | SfM init vs random100K | Test whether HCM0204 benefits from geometry prior or MCMC's canonical broad support. |
| 4 | Cap 4.5M vs 5.1M | Measure score/VRAM/size efficiency under the practical budget. |
| 5 | Noise `0`, `2.5e5`, `5e5` | Diagnose whether SGLD helps fine detail or adds instability. |
| 6 | Opacity/scale regularizers jointly `0` vs `0.01` | Confirm that the MCMC objective, not only relocation/growth, contributes. |
| 7 | Seeds 0, 1, 2 for the best two configurations | Report stochastic variance before selecting a final method. |

For every run, retain weighted score, SSIM, PSNR, LPIPS, 60-image size,
estimated 434-image size, peak allocated/reserved VRAM, wall time, final
Gaussian count, and the high-frequency/edge-region error used in the current
failure analysis. Do not run the expensive ablations until the 9K resource gate
has reached 5.1M without OOM and the split-resume smoke check has passed.

## Render and evaluation contract

Public evaluation uses the same post-processing ablation as Improved-GS:

```bash
python render_scene.py \
  --model_dir /kaggle/working/mcmc_models \
  --input_dir /kaggle/working/cleaned_mcmc_hcm0204 \
  --image_dir /kaggle/working/mcmc_render_variants \
  --orig_dir <public-set-root> \
  --scene_name HCM0204 \
  --iterations 30000 \
  --render_variants \
  --variant_sharpen_amount 0.3 \
  --sharpen_sigma 0.7 \
  --jpeg_quality 96 \
  --jpeg_subsampling 0 \
  --jpeg_optimize
```

This produces bilinear/bicubic crossed with sharpen 0/0.3. The primary result
is `bicubic_sharp0p3`; `bicubic_sharp0` is also reported so model quality is not
hidden by sharpening. Evaluation uses `eval_scene.py --psnr_max 40` and reports
the actual 60-image size plus an estimate for 434 images.

The Q96 Improved-GS reference is:

| Variant | SSIM | PSNR | LPIPS | Weighted score | Estimated 434 size |
|---|---:|---:|---:|---:|---:|
| bicubic, sharpen 0.3 | 0.846038 | 24.879372 | 0.083331 | 0.807075 | 312.612 MB |

Recalculate output size for MCMC. Sharper or more textured renders may compress
differently even with identical JPEG settings.

## Export policy

The default export contains lightweight reproducibility artifacts:

- `experiment_manifest.json`;
- `resume_manifest.json`;
- aggregate evaluation JSON and ranking JSON;
- MCMC statistics/logs; and
- a ZIP of the 60 images from the best public variant.

The final model bundle (`cfg_args` plus the final PLY) and resume bundle (latest
checkpoint plus manifest) are separate optional archives. Do not put a multi-GB
checkpoint in the render ZIP. Kaggle working storage is ephemeral; save a
notebook version with outputs or publish the resume bundle as a private Kaggle
dataset before ending a session.
