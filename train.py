#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import json
import math
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import (
    IMPROVED_GS_RASTERIZER_AVAILABLE,
    PIXEL_GS_RASTERIZER_AVAILABLE,
    network_gui,
    render,
)
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.improved_gs_utils import (
    build_improvedgs_resume_config,
    capture_improvedgs_runtime_state,
    compute_active_gaussian_budget,
    deterministic_eas_sample_indices,
    mu_update_interval,
    normalize_to_unit_range,
    prepare_edge_map_cache,
    rap_prune_iterations,
    restore_improvedgs_runtime_state,
    seed_everything,
    should_step_optimizer,
    validate_improvedgs_resume_config,
)
from utils.mcmc_utils import (
    build_mcmc_resume_config,
    capture_mcmc_runtime_state,
    restore_mcmc_runtime_state,
    validate_mcmc_initialization,
    validate_mcmc_options,
    validate_mcmc_resume_config,
    write_mcmc_config,
)

from lpipsPyTorch import lpips

import csv
import random
import torchvision
from pathlib import Path
from PIL import Image
from torchvision.transforms.functional import to_tensor

from render_scene import (
    camera_from_csv_row, load_distortion_params,
    load_undistorted_camera_params, redistort_and_crop,
)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


_IMPROVEDGS_CONFIG_KEYS = (
    "density_control", "use_las", "use_rap", "use_gc", "use_absgrad",
    "use_eas", "use_mu", "gaussian_budget", "improvedgs_grad_threshold",
    "min_opacity", "split_distance", "opacity_reduction",
    "budget_warmup_until_offset", "improvedgs_reset_max_opacity",
    "rap_initial_prune", "rap_initial_prune_iter",
    "rap_initial_prune_opacity", "rap_prune_ratio", "rap_prune_offset",
    "rap_rounds", "edge_sample_cams", "edge_mask_erosion",
    "mu_start_iter", "mu_interval", "mu_second_start_iter",
    "mu_second_interval", "densify_from_iter", "densify_until_iter",
    "densification_interval", "opacity_reset_interval",
)

_PIXELGS_CONFIG_KEYS = (
    "density_control", "pixelgs_depth_threshold", "densify_grad_threshold",
    "percent_dense", "densify_from_iter", "densify_until_iter",
    "densification_interval", "opacity_reset_interval",
)


def _validate_density_control_options(opt):
    """Validate density-control arguments without changing 3DGS defaults."""
    density_control = str(opt.density_control).lower()
    if density_control not in ("3dgs", "pixelgs", "improvedgs", "mcmc"):
        raise ValueError(
            "density_control must be one of '3dgs', 'pixelgs', 'improvedgs', or 'mcmc'"
        )
    opt.density_control = density_control

    component_names = ("use_las", "use_rap", "use_gc", "use_absgrad", "use_eas", "use_mu")
    for name in component_names:
        value = getattr(opt, name)
        if value not in (0, 1, False, True):
            raise ValueError("{} must be 0 or 1".format(name))

    if density_control == "pixelgs":
        pixelgs_depth_threshold = float(opt.pixelgs_depth_threshold)
        if (
            pixelgs_depth_threshold != pixelgs_depth_threshold
            or pixelgs_depth_threshold in (float("inf"), float("-inf"))
            or pixelgs_depth_threshold <= 0.0
        ):
            raise ValueError("pixelgs_depth_threshold must be a positive finite value")
        if int(opt.densification_interval) <= 0 or int(opt.opacity_reset_interval) <= 0:
            raise ValueError("densification and opacity-reset intervals must be positive")
        if int(opt.densify_until_iter) <= int(opt.densify_from_iter):
            raise ValueError("densify_until_iter must be greater than densify_from_iter")
        if float(opt.densify_grad_threshold) < 0.0:
            raise ValueError("densify_grad_threshold must be non-negative")

    if density_control == "mcmc":
        # The MCMC Markov-chain lifecycle is a complete, standalone density
        # controller. Normalize every Improved-GS component off so saved
        # metadata cannot suggest that a hybrid method was trained.
        for name in component_names:
            setattr(opt, name, 0)
        return

    if density_control != "improvedgs":
        # The model uses this flag to allocate an additional AbsGrad buffer.
        # Improved-GS components are ignored outside Improved-GS mode, including
        # that additional memory allocation.
        opt.use_absgrad = 0
        return
    if int(opt.gaussian_budget) <= 0:
        raise ValueError("gaussian_budget must be positive for ImprovedGS")
    if int(opt.densification_interval) <= 0 or int(opt.opacity_reset_interval) <= 0:
        raise ValueError("densification and opacity-reset intervals must be positive")
    if int(opt.densify_until_iter) <= int(opt.densify_from_iter):
        raise ValueError("densify_until_iter must be greater than densify_from_iter")
    if float(opt.improvedgs_grad_threshold) < 0.0:
        raise ValueError("improvedgs_grad_threshold must be non-negative")
    if not 0.0 <= float(opt.min_opacity) < 1.0:
        raise ValueError("min_opacity must be in [0,1)")
    if not 0.0 < float(opt.split_distance) < 1.0:
        raise ValueError("split_distance must be in (0,1)")
    if not 0.0 < float(opt.opacity_reduction) <= 1.0:
        raise ValueError("opacity_reduction must be in (0,1]")
    if int(opt.budget_warmup_until_offset) < 0:
        raise ValueError("budget_warmup_until_offset must be non-negative")
    if int(opt.edge_sample_cams) == 0 or int(opt.edge_sample_cams) < -1:
        raise ValueError("edge_sample_cams must be -1 or a positive integer")
    if int(opt.edge_mask_erosion) < 0:
        raise ValueError("edge_mask_erosion must be non-negative")
    if not 0.0 <= float(opt.rap_prune_ratio) < 1.0:
        raise ValueError("rap_prune_ratio must be in [0,1)")
    if (
        int(opt.rap_rounds) < 0
        or int(opt.rap_prune_offset) < 0
        or int(opt.rap_initial_prune_iter) < 0
    ):
        raise ValueError("RAP rounds, offset, and initial iteration must be non-negative")
    if not 0.0 <= float(opt.rap_initial_prune_opacity) < 1.0:
        raise ValueError("rap_initial_prune_opacity must be in [0,1)")
    if not 0.0 < float(opt.improvedgs_reset_max_opacity) < 1.0:
        raise ValueError("improvedgs_reset_max_opacity must be in (0,1)")
    if int(opt.mu_interval) <= 0 or int(opt.mu_second_interval) <= 0:
        raise ValueError("MU intervals must be positive")
    if int(opt.mu_start_iter) < 0:
        raise ValueError("mu_start_iter must be non-negative")
    if int(opt.mu_second_start_iter) <= int(opt.mu_start_iter):
        raise ValueError("mu_second_start_iter must be greater than mu_start_iter")
    if bool(opt.use_mu) and int(opt.mu_start_iter) < int(opt.densify_until_iter):
        raise ValueError(
            "mu_start_iter must be at or after densify_until_iter so a structural "
            "split cannot discard gradients accumulated by MU"
        )


def _write_improvedgs_config(dataset, opt, seed):
    """Persist the method configuration beside the legacy cfg_args snapshot."""
    config = {key: getattr(opt, key) for key in _IMPROVEDGS_CONFIG_KEYS}
    config["seed"] = int(seed)
    path = os.path.join(dataset.model_path, "improvedgs_config.json")
    with open(path, "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, indent=2, sort_keys=True)


def _build_pixelgs_config(opt, seed, cap_max):
    """Return the Pixel-GS settings that must match when training resumes."""
    config = {key: getattr(opt, key) for key in _PIXELGS_CONFIG_KEYS}
    config["cap_max"] = int(cap_max)
    config["seed"] = int(seed)
    return config


def _write_pixelgs_config(dataset, config):
    """Persist the paper-specific Pixel-GS configuration beside cfg_args."""
    path = os.path.join(dataset.model_path, "pixelgs_config.json")
    with open(path, "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, indent=2, sort_keys=True)


def _validate_pixelgs_resume_config(runtime_state, current_config):
    """Reject checkpoints produced with another density-control configuration."""
    saved_config = runtime_state.get("resume_config")
    if runtime_state.get("density_control") != "pixelgs" or saved_config is None:
        raise ValueError(
            "Pixel-GS resume requires a checkpoint created with "
            "--density_control pixelgs by this implementation"
        )
    if saved_config != current_config:
        differing_keys = [
            key
            for key in sorted(set(saved_config) | set(current_config))
            if saved_config.get(key) != current_config.get(key)
        ]
        raise ValueError(
            "Pixel-GS checkpoint configuration mismatch in: {}. Resume with "
            "the original Pixel-GS threshold, densification settings, cap, and "
            "seed.".format(", ".join(differing_keys))
        )


def _checkpoint_iteration(path):
    """Return the numeric suffix of ``chkpntN.pth`` or ``None``."""
    name = Path(path).name
    if not name.startswith("chkpnt") or not name.endswith(".pth"):
        return None
    suffix = name[len("chkpnt"):-len(".pth")]
    return int(suffix) if suffix.isdigit() else None


def _atomic_save_checkpoint(payload, path):
    """Write a checkpoint atomically so a failed save cannot corrupt resume."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = Path(str(path) + ".tmp")
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    except Exception:
        # A partial temporary file is never considered a resumable checkpoint.
        # Leave it in place for post-mortem inspection; the next save overwrites it.
        raise
    return path


def _prune_old_checkpoints(model_path, keep_last):
    """Remove older completed checkpoints after a newer atomic save succeeds."""
    keep_last = int(keep_last)
    if keep_last < 0:
        raise ValueError("checkpoint_keep_last must be non-negative")
    if keep_last == 0:
        return []

    candidates = []
    for path in Path(model_path).glob("chkpnt*.pth"):
        iteration = _checkpoint_iteration(path)
        if iteration is not None:
            candidates.append((iteration, path))
    candidates.sort(key=lambda item: item[0])
    removed = []
    for _, path in candidates[:-keep_last]:
        path.unlink()
        removed.append(path)
    return removed


def _visibility_as_bool(visibility_filter, num_gaussians, device):
    """Normalize renderer boolean masks or index tensors to a flat bool mask."""
    if visibility_filter.dtype == torch.bool:
        mask = visibility_filter.reshape(-1)
        if mask.numel() != num_gaussians:
            raise RuntimeError("Renderer visibility mask has the wrong length")
        return mask.to(device=device)
    indices = visibility_filter.reshape(-1).long().to(device=device)
    mask = torch.zeros((num_gaussians,), dtype=torch.bool, device=device)
    if indices.numel():
        mask[indices] = True
    return mask


def _compute_eas_scores(
    cameras,
    edge_maps,
    iteration,
    opt,
    gaussians,
    pipe,
    bg,
    dataset,
):
    """Render sampled edge-weighted views and accumulate per-Gaussian EAS."""
    if not cameras or not edge_maps:
        raise RuntimeError("EAS is enabled but no training edge maps are available")
    sample_all = (
        int(opt.edge_sample_cams) == -1
        or (
            iteration % int(opt.opacity_reset_interval) == 400
            and iteration < 9_000
        )
    )
    sample_count = -1 if sample_all else int(opt.edge_sample_cams)
    sample_indices = deterministic_eas_sample_indices(
        total_cameras=len(cameras),
        sample_count=sample_count,
        iteration=int(iteration),
        densify_from_iter=int(opt.densify_from_iter),
        densification_interval=int(opt.densification_interval),
    )
    num_gaussians = int(gaussians.get_xyz.shape[0])
    scores = torch.zeros((num_gaussians,), device=gaussians.get_xyz.device, dtype=torch.float32)
    any_visible = False

    with torch.no_grad():
        for camera_index in sample_indices:
            pixel_weights = edge_maps[camera_index].to(
                device=gaussians.get_xyz.device,
                dtype=torch.float32,
                non_blocking=True,
            )
            try:
                eas_pkg = render(
                    cameras[camera_index], gaussians, pipe, bg,
                    use_trained_exp=dataset.train_test_exp,
                    separate_sh=SPARSE_ADAM_AVAILABLE,
                    track_gradients=False,
                    pixel_weights=pixel_weights,
                )
            except TypeError as error:
                raise RuntimeError(
                    "EAS requires the ImprovedGS renderer interface "
                    "render(..., track_gradients=False, pixel_weights=...)."
                ) from error
            accum_weights = eas_pkg.get("accum_weights")
            if accum_weights is None:
                raise RuntimeError(
                    "EAS is enabled, but the rasterizer did not return accum_weights."
                )
            accum_weights = accum_weights.detach().reshape(-1)
            if accum_weights.numel() != num_gaussians:
                raise RuntimeError(
                    "accum_weights must contain one value per Gaussian ({} != {}).".format(
                        accum_weights.numel(), num_gaussians
                    )
                )
            visible = _visibility_as_bool(
                eas_pkg["visibility_filter"], num_gaussians, scores.device
            )
            if visible.any():
                normalized = normalize_to_unit_range(accum_weights).to(scores.device)
                scores[visible] += normalized[visible] / float(len(sample_indices))
                any_visible = True

    return (scores if any_visible else None), len(sample_indices)

# test evaluation
def load_gt_image(gt_dir, image_name, device):
    """
    GT nam trong test/images/, cung TEN GOC nhung duoi co the khac
    (vd .JPG thay vi .jpg trong CSV). Khop theo STEM, thu nhieu bien
    the duoi file khong phan biet hoa/thuong.
    """
    stem = Path(image_name).stem
    gt_dir = Path(gt_dir)

    for ext in (".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"):
        gt_path = gt_dir / f"{stem}{ext}"
        if gt_path.exists():
            img = Image.open(gt_path).convert("RGB")
            return to_tensor(img).to(device)

    return None


def render_test_samples(dataset, gaussians, pipe, background, iteration,
                         orig_dir, num_samples=15, seed=42, analyse_file=None):
    """
    Render mot mau co dinh (seed) tu test_poses.csv, redistort+crop ve dung
    kich thuoc GT goc, so sanh voi GT that (test/images/*.jpg), ghi metric
    ra analyse_file.
    """
    source_path = Path(dataset.source_path)
    scene_name = source_path.parent.name
    input_dir = source_path.parent.parent

    scene_dir = input_dir / scene_name  
    test_poses_csv = scene_dir / "test" / "test_poses.csv"
    gt_dir = scene_dir / "test" / "images"

    if not test_poses_csv.exists():
        print(f"[TEST RENDER] Khong tim thay {test_poses_csv}, bo qua test render.", flush=True)
        return

    dist = load_distortion_params(orig_dir, scene_name)
    und = load_undistorted_camera_params(input_dir, scene_name)

    with open(test_poses_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return

    rng = random.Random(seed)
    sample_rows = rng.sample(rows, min(num_samples, len(rows)))

    out_dir = Path(dataset.model_path) / "test_renders" / f"iter_{iteration}"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = dataset.data_device
    l1_sum = ssim_sum = psnr_sum = lpips_sum = 0.0
    n_scored = 0

    with torch.no_grad():
        for idx, row in enumerate(sample_rows):
            camera = camera_from_csv_row(
                row, idx, device,
                width=und["width"], height=und["height"],
                fx=und["f"], fy=und["f"],
            )
            rendering = render(
                camera, gaussians, pipe, background,
                use_trained_exp=dataset.train_test_exp,
                separate_sh=SPARSE_ADAM_AVAILABLE,
            )["render"]

            if abs(dist["k"]) > 1e-8:
                rendering = redistort_and_crop(
                    rendering,
                    f=und["f"], cx_render=und["cx"], cy_render=und["cy"],
                    k=dist["k"], cx_orig=dist["cx"], cy_orig=dist["cy"],
                    orig_w=dist["width"], orig_h=dist["height"],
                )

            torchvision.utils.save_image(rendering, out_dir / Path(row["image_name"]).name)

            gt_image = load_gt_image(gt_dir, row["image_name"], device)
            if gt_image is not None:
                image_c = torch.clamp(rendering, 0.0, 1.0)
                gt_image_c = torch.clamp(gt_image, 0.0, 1.0)
                if image_c.shape != gt_image_c.shape:
                    print(f"[TEST RENDER] Bo qua {row['image_name']}: "
                          f"shape render {tuple(image_c.shape)} != GT "
                          f"{tuple(gt_image_c.shape)}", flush=True)
                else:
                    l1_val = l1_loss(image_c, gt_image_c).item()
                    ssim_val = ssim(image_c, gt_image_c).item()
                    psnr_val = psnr(image_c, gt_image_c).mean().item()
                    lpips_val = lpips(image_c.unsqueeze(0), gt_image_c.unsqueeze(0),
                                       net_type='squeeze').item()
                    psnr_norm = torch.clamp(torch.tensor(psnr_val / 40.0), 0.0, 1.0).item()
                    score = 0.4 * (1 - lpips_val) + 0.3 * ssim_val + 0.3 * psnr_norm

                    l1_sum += l1_val; ssim_sum += ssim_val
                    psnr_sum += psnr_val; lpips_sum += lpips_val
                    n_scored += 1

            del camera, rendering

    print(f"[TEST RENDER] Saved {len(sample_rows)} renders @ iter {iteration} -> {out_dir}", flush=True)

    if n_scored > 0:
        l1_avg = l1_sum / n_scored
        ssim_avg = ssim_sum / n_scored
        psnr_avg = psnr_sum / n_scored
        lpips_avg = lpips_sum / n_scored
        psnr_norm_avg = min(max(psnr_avg / 40.0, 0.0), 1.0)
        score_avg = 0.4 * (1 - lpips_avg) + 0.3 * ssim_avg + 0.3 * psnr_norm_avg

        print(f"[TEST ITER {iteration}] n={n_scored} L1={l1_avg:.4f} "
              f"SSIM={ssim_avg:.4f} LPIPS={lpips_avg:.4f} PSNR={psnr_avg:.2f} "
              f"score={score_avg:.4f}", flush=True)

        if analyse_file is not None:
            analyse_file.write(f"{iteration},{l1_avg:.6f},{ssim_avg:.6f},"
                                f"{lpips_avg:.6f},{psnr_avg:.6f},{score_avg:.6f}\n")
            analyse_file.flush()
    else:
        print(f"[TEST ITER {iteration}] Khong tim thay GT nao khop "
              f"(kiem tra gt_dir: {gt_dir})", flush=True)

# VAR: add cap max
def training(
    dataset,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    cap_max=-1,
    analyse_path=None,
    orig_dir=None,
    test_render_every=50,
    test_render_samples=15,
    seed=0,
    stop_after_iteration=-1,
    checkpoint_keep_last=0,
    stats_path=None,
):

    _validate_density_control_options(opt)
    improved_mode = opt.density_control == "improvedgs"
    pixelgs_mode = opt.density_control == "pixelgs"
    mcmc_mode = opt.density_control == "mcmc"

    stop_after_iteration = int(stop_after_iteration)
    if stop_after_iteration == -1:
        run_until_iteration = int(opt.iterations)
    elif 0 < stop_after_iteration <= int(opt.iterations):
        run_until_iteration = stop_after_iteration
    else:
        raise ValueError(
            "stop_after_iteration must be -1 or in [1, iterations]"
        )
    checkpoint_keep_last = int(checkpoint_keep_last)
    if checkpoint_keep_last < 0:
        raise ValueError("checkpoint_keep_last must be non-negative")
    if mcmc_mode:
        validate_mcmc_options(opt, int(cap_max))
        validate_mcmc_initialization(dataset)
        if (
            str(dataset.mcmc_init_type).lower() == "random"
            and int(dataset.mcmc_random_points) > int(cap_max)
        ):
            raise ValueError(
                "mcmc_random_points ({}) cannot exceed cap_max ({})".format(
                    dataset.mcmc_random_points, cap_max
                )
            )
        # Scene keeps every legacy method on the ordinary SfM loader. These
        # transient fields opt MCMC into its explicit random/SfM initializer.
        dataset.mcmc_enabled = True
        dataset.mcmc_init_seed = int(seed)

    if (
        improved_mode
        and (bool(opt.use_absgrad) or bool(opt.use_eas))
        and not IMPROVED_GS_RASTERIZER_AVAILABLE
    ):
        raise RuntimeError(
            "AbsGrad/EAS requires the tracked Improved-GS rasterizer patch. "
            "Run `python scripts/apply_improved_gs_rasterizer_patch.py` and "
            "force-reinstall submodules/diff-gaussian-rasterization."
        )

    if pixelgs_mode and not PIXEL_GS_RASTERIZER_AVAILABLE:
        raise RuntimeError(
            "Pixel-GS requires the tracked rasterizer patch. Run "
            "`python scripts/apply_improved_gs_rasterizer_patch.py` and "
            "force-reinstall submodules/diff-gaussian-rasterization."
        )

    if mcmc_mode:
        try:
            from diff_gaussian_rasterization import compute_relocation as _compute_relocation
        except (ImportError, AttributeError) as error:
            raise RuntimeError(
                "MCMC requires the tracked rasterizer patch with the "
                "compute_relocation CUDA symbol. Apply the project patch and "
                "force-reinstall submodules/diff-gaussian-rasterization."
            ) from error
        del _compute_relocation

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")
    if improved_mode and bool(opt.use_mu) and opt.optimizer_type == "sparse_adam":
        raise ValueError(
            "ImprovedGS MU accumulates gradients across multiple views and is not "
            "safe with sparse_adam visibility masks. Use --optimizer_type default "
            "or disable MU with --use_mu 0."
        )

    first_iter = 0
    checkpoint_runtime_state = None
    tb_writer = prepare_output_and_logger(dataset, write_config=False)
    improved_resume_config = (
        build_improvedgs_resume_config(dataset, opt, pipe, seed)
        if improved_mode
        else None
    )
    pixelgs_resume_config = (
        _build_pixelgs_config(opt, seed, cap_max)
        if pixelgs_mode
        else None
    )
    mcmc_resume_config = (
        build_mcmc_resume_config(dataset, opt, pipe, seed, int(cap_max))
        if mcmc_mode
        else None
    )
    if improved_mode:
        if cap_max > 0:
            print(
                "[ImprovedGS] --cap_max is ignored; using the strict "
                "--gaussian_budget {}.".format(opt.gaussian_budget),
                flush=True,
            )

    #VAR: add logger
    analyse_file = None
    if analyse_path is not None:
        out_dir = os.path.dirname(analyse_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        analyse_file = open(analyse_path, "w")
        analyse_file.write("iteration,L1,SSIM,LPIPS,PSNR,score\n")

    gaussians = GaussianModel(
        dataset.sh_degree,
        opt.optimizer_type,
        mcmc_init_mode=(opt.mcmc_init_mode if mcmc_mode else "legacy"),
    )
    scene = Scene(dataset, gaussians)
    pixelgs_absolute_depth_threshold = (
        float(opt.pixelgs_depth_threshold) * float(scene.cameras_extent)
        if pixelgs_mode
        else None
    )
    if pixelgs_mode and (
        not math.isfinite(pixelgs_absolute_depth_threshold)
        or pixelgs_absolute_depth_threshold <= 0.0
    ):
        raise ValueError("Pixel-GS requires a positive scene camera extent")
    gaussians.training_setup(opt)
    if checkpoint:
        checkpoint_payload = torch.load(checkpoint, weights_only=False)
        if not isinstance(checkpoint_payload, (tuple, list)):
            raise ValueError("Unsupported checkpoint payload")
        if len(checkpoint_payload) == 2:
            model_params, first_iter = checkpoint_payload
        elif len(checkpoint_payload) == 3:
            model_params, first_iter, checkpoint_runtime_state = checkpoint_payload
        else:
            raise ValueError(
                "Unsupported checkpoint payload ({} entries)".format(
                    len(checkpoint_payload)
                )
            )
        gaussians.restore(model_params, opt)

    if improved_mode and int(gaussians.get_xyz.shape[0]) > int(opt.gaussian_budget):
        raise ValueError(
            "The initialized/checkpoint model contains {} Gaussians, exceeding "
            "the strict final budget {}. Increase --gaussian_budget or start "
            "from a smaller model.".format(
                gaussians.get_xyz.shape[0], opt.gaussian_budget
            )
        )

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    all_train_cameras = scene.getTrainCameras().copy()
    if checkpoint_runtime_state is not None:
        if pixelgs_mode:
            _validate_pixelgs_resume_config(
                checkpoint_runtime_state, pixelgs_resume_config
            )
        elif mcmc_mode:
            validate_mcmc_resume_config(
                checkpoint_runtime_state, mcmc_resume_config
            )
        elif not improved_mode:
            raise ValueError(
                "This checkpoint contains method-specific runtime state but the "
                "current run uses --density_control {}".format(opt.density_control)
            )
        else:
            validate_improvedgs_resume_config(
                checkpoint_runtime_state, improved_resume_config
            )
            if checkpoint_runtime_state.get("resume_config") is None:
                print(
                    "[ImprovedGS] Checkpoint has no saved method configuration; "
                    "the caller is responsible for using the original flags.",
                    flush=True,
                )

        if improved_mode or mcmc_mode:
            current_cameras_by_name = {
                camera.image_name: camera for camera in all_train_cameras
            }
            if len(current_cameras_by_name) != len(all_train_cameras):
                raise ValueError("Training camera image names must be unique")
            saved_camera_order = checkpoint_runtime_state.get("camera_order_names")
            if saved_camera_order is not None:
                if (
                    len(saved_camera_order) != len(all_train_cameras)
                    or len(set(saved_camera_order)) != len(saved_camera_order)
                    or set(saved_camera_order) != set(current_cameras_by_name)
                ):
                    raise ValueError(
                        "Checkpoint camera set does not match the current scene"
                    )
                all_train_cameras = [
                    current_cameras_by_name[name] for name in saved_camera_order
                ]

            saved_exposure_mapping = checkpoint_runtime_state.get("exposure_mapping")
            if saved_exposure_mapping is not None and set(saved_exposure_mapping) != set(
                current_cameras_by_name
            ):
                raise ValueError(
                    "Checkpoint exposure mapping does not match the current scene"
                )

    viewpoint_indices = list(range(len(all_train_cameras)))
    viewpoint_stack = all_train_cameras.copy()
    eas_cameras = (
        all_train_cameras.copy()
        if improved_mode and bool(opt.use_eas)
        else []
    )
    eas_edge_maps = (
        prepare_edge_map_cache(eas_cameras, int(opt.edge_mask_erosion))
        if eas_cameras else []
    )
    scheduled_rap_prunes = set(
        rap_prune_iterations(
            int(opt.densify_from_iter),
            int(opt.densify_until_iter),
            int(opt.opacity_reset_interval),
            int(opt.rap_rounds),
            int(opt.rap_prune_offset),
        )
    ) if improved_mode and bool(opt.use_rap) else set()

    if checkpoint_runtime_state is not None and (improved_mode or mcmc_mode):
        remaining_names = checkpoint_runtime_state.get("remaining_camera_names")
        if remaining_names is not None:
            current_name_to_index = {
                camera.image_name: index
                for index, camera in enumerate(all_train_cameras)
            }
            if len(current_name_to_index) != len(all_train_cameras):
                raise ValueError("Training camera image names must be unique")
            if len(set(remaining_names)) != len(remaining_names) or any(
                name not in current_name_to_index for name in remaining_names
            ):
                raise ValueError("Checkpoint contains invalid remaining camera names")
            saved_indices = [current_name_to_index[name] for name in remaining_names]
        else:
            saved_indices = [
                int(index) for index in checkpoint_runtime_state["viewpoint_indices"]
            ]
            if (
                len(set(saved_indices)) != len(saved_indices)
                or any(
                    index < 0 or index >= len(all_train_cameras)
                    for index in saved_indices
                )
            ):
                raise ValueError("Checkpoint contains invalid remaining camera indices")
        viewpoint_indices = saved_indices
        viewpoint_stack = [all_train_cameras[index] for index in saved_indices]
        if mcmc_mode:
            restore_mcmc_runtime_state(gaussians, checkpoint_runtime_state)
        else:
            restore_improvedgs_runtime_state(gaussians, checkpoint_runtime_state)
    elif checkpoint and (pixelgs_mode or mcmc_mode) and checkpoint_runtime_state is None:
        raise ValueError(
            "Cannot verify {} settings or restore its stochastic runtime state "
            "from this legacy checkpoint. Resume from a checkpoint created by "
            "this implementation.".format(opt.density_control)
        )
    elif checkpoint and improved_mode:
        interval = mu_update_interval(
            int(first_iter),
            use_mu=bool(opt.use_mu),
            first_stage_start=int(opt.mu_start_iter),
            second_stage_start=int(opt.mu_second_start_iter),
            first_stage_interval=int(opt.mu_interval),
            second_stage_interval=int(opt.mu_second_interval),
        )
        if interval > 1 and int(first_iter) % interval != 0:
            raise ValueError(
                "Legacy checkpoint iteration {} falls inside a {}-view MU "
                "accumulation window, but legacy checkpoints do not store "
                "pending gradients. Resume from an optimizer-boundary "
                "checkpoint instead.".format(first_iter, interval)
            )
        print(
            "[ImprovedGS] Loading a legacy checkpoint without RNG/camera "
            "runtime state; resume is supported but not bitwise reproducible.",
            flush=True,
        )

    if int(first_iter) >= run_until_iteration:
        raise ValueError(
            "Checkpoint iteration {} is not before this stage target {}".format(
                first_iter, run_until_iteration
            )
        )
    if mcmc_mode and int(gaussians.get_xyz.shape[0]) > int(cap_max):
        raise ValueError(
            "The initialized/checkpoint model contains {} Gaussians, exceeding "
            "the MCMC cap {}.".format(gaussians.get_xyz.shape[0], cap_max)
        )

    if improved_mode:
        # Write metadata only after a resumed checkpoint has passed all
        # compatibility checks, so a failed resume cannot overwrite the
        # original run configuration.
        _write_improvedgs_config(dataset, opt, seed)
    elif pixelgs_mode:
        _write_pixelgs_config(dataset, pixelgs_resume_config)
    elif mcmc_mode:
        write_mcmc_config(dataset, mcmc_resume_config)
    _write_cfg_args(dataset)

    stats_file = None
    if stats_path is not None:
        if not mcmc_mode:
            raise ValueError("stats_path is currently supported only for MCMC")
        stats_path = Path(stats_path)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_file = open(
            stats_path,
            "a" if checkpoint else "w",
            encoding="utf-8",
            buffering=1,
        )
        stats_file.write(
            json.dumps(
                {
                    "event": "run_start",
                    "first_iteration": int(first_iter),
                    "run_until_iteration": int(run_until_iteration),
                    "total_iterations": int(opt.iterations),
                    "cap_max": int(cap_max),
                    "seed": int(seed),
                },
                sort_keys=True,
            )
            + "\n"
        )
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(
        range(first_iter, run_until_iteration), desc="Training progress"
    )
    first_iter += 1
    for iteration in range(first_iter, run_until_iteration + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        xyz_lr = gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = all_train_cameras.copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        if pixelgs_mode and iteration < int(opt.densify_until_iter):
            render_pkg = render(
                viewpoint_cam,
                gaussians,
                pipe,
                bg,
                use_trained_exp=dataset.train_test_exp,
                separate_sh=SPARSE_ADAM_AVAILABLE,
                track_gradients=True,
                track_pixel_counts=True,
                pixelgs_depth_threshold=pixelgs_absolute_depth_threshold,
            )
        elif improved_mode and bool(opt.use_absgrad):
            try:
                render_pkg = render(
                    viewpoint_cam, gaussians, pipe, bg,
                    use_trained_exp=dataset.train_test_exp,
                    separate_sh=SPARSE_ADAM_AVAILABLE,
                    track_gradients=True,
                )
            except TypeError as error:
                raise RuntimeError(
                    "AbsGrad requires the ImprovedGS renderer interface "
                    "render(..., track_gradients=True)."
                ) from error
        else:
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        mcmc_opacity_penalty = 0.0
        mcmc_scale_penalty = 0.0
        if mcmc_mode:
            mcmc_opacity_penalty = (
                float(opt.mcmc_opacity_reg) * gaussians.get_opacity.abs().mean()
            )
            mcmc_scale_penalty = (
                float(opt.mcmc_scale_reg) * gaussians.get_scaling.abs().mean()
            )
            loss = loss + mcmc_opacity_penalty + mcmc_scale_penalty

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            # VAR: render mau test that (co GT) va ghi metric
            if orig_dir is not None and analyse_file is not None and iteration % test_render_every == 0:
                render_test_samples(
                    dataset, gaussians, pipe, background, iteration,
                    orig_dir, num_samples=test_render_samples,
                    analyse_file=analyse_file,
                )

            if iteration % 10 == 0:
                # VAR: log gaussians numbers 
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}", "N": f"{gaussians.get_xyz.shape[0]}"})
                progress_bar.update(10)
            if iteration == run_until_iteration:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if improved_mode:
                # ImprovedGS applies parameter updates before structural changes.
                # With MU disabled (or before 15k), this is one update per view;
                # later, gradients remain untouched until the scheduled 5/20-view
                # accumulation boundary.
                if should_step_optimizer(
                    iteration,
                    int(opt.iterations),
                    use_mu=bool(opt.use_mu),
                    first_stage_start=int(opt.mu_start_iter),
                    second_stage_start=int(opt.mu_second_start_iter),
                    first_stage_interval=int(opt.mu_interval),
                    second_stage_interval=int(opt.mu_second_interval),
                ):
                    gaussians.exposure_optimizer.step()
                    gaussians.exposure_optimizer.zero_grad(set_to_none=True)
                    if use_sparse_adam:
                        visible = radii > 0
                        gaussians.optimizer.step(visible, radii.shape[0])
                        gaussians.optimizer.zero_grad(set_to_none=True)
                    else:
                        gaussians.optimizer.step()
                        gaussians.optimizer.zero_grad(set_to_none=True)

                if bool(opt.use_rap) and bool(opt.rap_initial_prune) and iteration == int(opt.rap_initial_prune_iter):
                    initial_pruned = gaussians.only_prune(float(opt.rap_initial_prune_opacity))
                    tqdm.write(
                        "[ImprovedGS RAP @ {}] initial opacity prune: -{}".format(
                            iteration, initial_pruned
                        )
                    )

                if opt.densify_from_iter < iteration < opt.densify_until_iter:
                    # Keep radius statistics for model bookkeeping, but ImprovedGS
                    # deliberately does not use the original large-Gaussian prune.
                    gaussians.max_radii2D[visibility_filter] = torch.max(
                        gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                    )
                    if bool(opt.use_absgrad):
                        gaussians.add_densification_stats_abs(
                            viewspace_point_tensor, visibility_filter
                        )
                    else:
                        gaussians.add_densification_stats(
                            viewspace_point_tensor, visibility_filter
                        )

                    if iteration % opt.densification_interval == 0:
                        eas_scores = None
                        eas_view_count = 0
                        if bool(opt.use_eas):
                            eas_scores, eas_view_count = _compute_eas_scores(
                                eas_cameras,
                                eas_edge_maps,
                                iteration,
                                opt,
                                gaussians,
                                pipe,
                                bg,
                                dataset,
                            )
                        active_budget = compute_active_gaussian_budget(
                            iteration=int(iteration),
                            densify_from_iter=int(opt.densify_from_iter),
                            densify_until_iter=int(opt.densify_until_iter),
                            final_budget=int(opt.gaussian_budget),
                            use_growth_control=bool(opt.use_gc),
                            warmup_until_offset=int(opt.budget_warmup_until_offset),
                        )
                        density_report = gaussians.densify_and_prune_improved(
                            eas_scores,
                            float(opt.min_opacity),
                            active_budget,
                            opt,
                            iteration,
                            scene.cameras_extent,
                        )
                        tqdm.write(
                            "[ImprovedGS @ {iteration}] N {before}->{after}, "
                            "split={split}, opacity_pruned={pruned}, budget={budget}, "
                            "EAS_views={eas_views}".format(
                                eas_views=eas_view_count,
                                **density_report,
                            )
                        )

                if bool(opt.use_rap):
                    if iteration in scheduled_rap_prunes:
                        rap_pruned = gaussians.only_prune(
                            float(opt.rap_prune_ratio), percent=True
                        )
                        tqdm.write(
                            "[ImprovedGS RAP @ {}] recovery prune: -{} ({:.1%})".format(
                                iteration, rap_pruned, float(opt.rap_prune_ratio)
                            )
                        )
                    if (
                        opt.densify_from_iter < iteration < opt.densify_until_iter
                        and iteration % opt.opacity_reset_interval == 0
                    ):
                        gaussians.reset_opacity(float(opt.improvedgs_reset_max_opacity))
                elif iteration < opt.densify_until_iter and (
                    iteration % opt.opacity_reset_interval == 0
                    or (dataset.white_background and iteration == opt.densify_from_iter)
                ):
                    gaussians.reset_opacity()
            elif mcmc_mode:
                mcmc_density_report = None
                if (
                    int(opt.densify_from_iter) < iteration
                    < int(opt.densify_until_iter)
                    and iteration % int(opt.densification_interval) == 0
                ):
                    mcmc_density_report = gaussians.mcmc_relocate_and_grow(
                        cap_max=int(cap_max),
                        min_opacity=float(opt.mcmc_min_opacity),
                        growth_rate=float(opt.mcmc_growth_rate),
                    )
                    tqdm.write(
                        "[MCMC @ {iteration}] N {before}->{after}, "
                        "relocated={relocated}, added={added}, cap={cap_max}".format(
                            iteration=iteration, **mcmc_density_report
                        )
                    )

                # Match the official MCMC lifecycle: structural Markov moves,
                # then Adam, then covariance-shaped SGLD on xyz only.
                if iteration < int(opt.iterations):
                    gaussians.exposure_optimizer.step()
                    gaussians.exposure_optimizer.zero_grad(set_to_none=True)
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)
                    gaussians.add_mcmc_position_noise(
                        noise_lr=float(opt.mcmc_noise_lr),
                        xyz_lr=float(xyz_lr),
                        min_opacity=float(opt.mcmc_min_opacity),
                        chunk_size=int(opt.mcmc_noise_chunk_size),
                    )

                if stats_file is not None and (
                    mcmc_density_report is not None
                    or iteration % int(opt.densification_interval) == 0
                    or iteration == run_until_iteration
                ):
                    stats_record = {
                        "event": (
                            "density_control"
                            if mcmc_density_report is not None
                            else "iteration"
                        ),
                        "iteration": int(iteration),
                        "gaussians": int(gaussians.get_xyz.shape[0]),
                        "loss": float(loss.item()),
                        "l1": float(Ll1.item()),
                        "ssim": float(ssim_value.item()),
                        "depth_loss": float(Ll1depth),
                        "opacity_penalty": float(mcmc_opacity_penalty.item()),
                        "scale_penalty": float(mcmc_scale_penalty.item()),
                        "xyz_lr": float(xyz_lr),
                        "peak_vram_allocated_bytes": int(
                            torch.cuda.max_memory_allocated()
                        ),
                        "peak_vram_reserved_bytes": int(
                            torch.cuda.max_memory_reserved()
                        ),
                    }
                    if mcmc_density_report is not None:
                        stats_record.update(mcmc_density_report)
                    stats_file.write(json.dumps(stats_record, sort_keys=True) + "\n")
            else:
                # Densification
                if iteration < opt.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    if pixelgs_mode:
                        pixel_counts = render_pkg.get("pixel_counts")
                        pixelgs_grad_scale = render_pkg.get("pixelgs_grad_scale")
                        if pixel_counts is None or pixelgs_grad_scale is None:
                            raise RuntimeError(
                                "Pixel-GS render must return pixel_counts and "
                                "pixelgs_grad_scale before densification ends"
                            )
                        gaussians.add_densification_stats_pixelgs(
                            viewspace_point_tensor,
                            visibility_filter,
                            pixel_counts,
                            pixelgs_grad_scale,
                        )
                    else:
                        gaussians.add_densification_stats(
                            viewspace_point_tensor, visibility_filter
                        )

                    # VAR: add capmax constraint
                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        if cap_max <= 0 or gaussians.get_xyz.shape[0] < cap_max:
                            size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                            gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                        else:
                            gaussians.tmp_radii = radii
                            prune_mask = (gaussians.get_opacity < 0.005).squeeze()
                            gaussians.prune_points(prune_mask)
                            gaussians.tmp_radii = None
                            torch.cuda.empty_cache()

                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()

                # Optimizer step
                if iteration < opt.iterations:
                    gaussians.exposure_optimizer.step()
                    gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                    if use_sparse_adam:
                        visible = radii > 0
                        gaussians.optimizer.step(visible, radii.shape[0])
                        gaussians.optimizer.zero_grad(set_to_none = True)
                    else:
                        gaussians.optimizer.step()
                        gaussians.optimizer.zero_grad(set_to_none = True)

            checkpoint_due = iteration in checkpoint_iterations or (
                mcmc_mode
                and iteration == run_until_iteration
                and run_until_iteration < int(opt.iterations)
            )
            if checkpoint_due:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                if improved_mode:
                    runtime_state = capture_improvedgs_runtime_state(
                        gaussians,
                        viewpoint_indices,
                        remaining_camera_names=[
                            all_train_cameras[index].image_name
                            for index in viewpoint_indices
                        ],
                        camera_order_names=[
                            camera.image_name for camera in all_train_cameras
                        ],
                    )
                    runtime_state["resume_config"] = improved_resume_config
                    checkpoint_payload = (
                        gaussians.capture(), iteration, runtime_state
                    )
                elif mcmc_mode:
                    runtime_state = capture_mcmc_runtime_state(
                        gaussians,
                        viewpoint_indices,
                        remaining_camera_names=[
                            all_train_cameras[index].image_name
                            for index in viewpoint_indices
                        ],
                        camera_order_names=[
                            camera.image_name for camera in all_train_cameras
                        ],
                        # The final iteration cannot be resumed further and its
                        # dense pending gradients would add roughly another
                        # model copy to the checkpoint.
                        include_gradients=(iteration < int(opt.iterations)),
                    )
                    runtime_state["resume_config"] = mcmc_resume_config
                    checkpoint_payload = (
                        gaussians.capture(), iteration, runtime_state
                    )
                elif pixelgs_mode:
                    checkpoint_payload = (
                        gaussians.capture(),
                        iteration,
                        {
                            "density_control": "pixelgs",
                            "resume_config": pixelgs_resume_config,
                        },
                    )
                else:
                    checkpoint_payload = (gaussians.capture(), iteration)
                checkpoint_path = _atomic_save_checkpoint(
                    checkpoint_payload,
                    Path(scene.model_path) / ("chkpnt" + str(iteration) + ".pth"),
                )
                removed_checkpoints = _prune_old_checkpoints(
                    scene.model_path, checkpoint_keep_last
                )
                if removed_checkpoints:
                    print(
                        "[Checkpoint retention] kept newest {}, removed {}".format(
                            checkpoint_keep_last, len(removed_checkpoints)
                        ),
                        flush=True,
                    )
                print("[Checkpoint] {}".format(checkpoint_path), flush=True)
    if analyse_file is not None:
        analyse_file.close()
    if stats_file is not None:
        stats_file.write(
            json.dumps(
                {
                    "event": "run_end",
                    "iteration": int(run_until_iteration),
                    "gaussians": int(gaussians.get_xyz.shape[0]),
                    "peak_vram_allocated_bytes": int(
                        torch.cuda.max_memory_allocated()
                    ),
                    "peak_vram_reserved_bytes": int(
                        torch.cuda.max_memory_reserved()
                    ),
                },
                sort_keys=True,
            )
            + "\n"
        )
        stats_file.close()

def _write_cfg_args(args):
    """Atomically update cfg_args only after resume compatibility succeeds."""
    path = Path(args.model_path) / "cfg_args"
    temporary_path = Path(str(path) + ".tmp")
    with open(temporary_path, "w", encoding="utf-8") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    os.replace(temporary_path, path)


def prepare_output_and_logger(args, write_config=True):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    if write_config:
        _write_cfg_args(args)

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument(
        "--stop_after_iteration",
        type=int,
        default=-1,
        help=(
            "Stop this process at an intermediate iteration without changing "
            "the full optimization schedule."
        ),
    )
    parser.add_argument(
        "--checkpoint_keep_last",
        type=int,
        default=0,
        help="Keep only the newest N completed checkpoints; 0 keeps all.",
    )
    parser.add_argument(
        "--stats_path",
        type=str,
        default=None,
        help="Optional JSONL output for MCMC training telemetry.",
    )
    parser.add_argument("--cap_max", type=int, default=-1)
    parser.add_argument("--analyse", type=str, default=None, help="Đường dẫn file log điểm số. Không truyền = không log.")
    parser.add_argument("--orig_dir", type=str, default=None,
        help="Duong dan chua cameras.bin SIMPLE_RADIAL goc (truoc undistort), can de redistort anh test.")
    parser.add_argument("--test_render_every", type=int, default=50)
    parser.add_argument("--test_render_samples", type=int, default=15)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    seed_everything(args.seed)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        args.cap_max,
        args.analyse,
        args.orig_dir,
        args.test_render_every,
        args.test_render_samples,
        seed=args.seed,
        stop_after_iteration=args.stop_after_iteration,
        checkpoint_keep_last=args.checkpoint_keep_last,
        stats_path=args.stats_path,
    )
    # All done
    print("\nTraining complete.")
