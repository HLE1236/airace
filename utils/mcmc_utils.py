"""Small, testable helpers for the independent 3DGS-MCMC training path.

The CUDA relocation primitive is imported lazily so this module remains usable
by CPU-only unit tests and by setup code before the rasterizer is compiled.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import platform
import random
from pathlib import Path

import numpy as np
import torch


MCMC_RELOCATION_N_MAX = 51


def _finite_float(value, name: str) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError("{} must be a real number".format(name)) from error
    if not math.isfinite(value):
        raise ValueError("{} must be finite".format(name))
    return value


def validate_mcmc_options(opt, cap_max: int) -> None:
    """Validate the coupled MCMC configuration without mutating ``opt``."""
    if isinstance(cap_max, bool) or not isinstance(cap_max, int):
        raise TypeError("cap_max must be an integer")
    if cap_max <= 0:
        raise ValueError("cap_max must be positive for MCMC")

    init_mode = str(getattr(opt, "mcmc_init_mode", "paper")).lower()
    if init_mode not in ("paper", "legacy"):
        raise ValueError("mcmc_init_mode must be 'paper' or 'legacy'")

    noise_lr = _finite_float(getattr(opt, "mcmc_noise_lr", 500_000.0), "mcmc_noise_lr")
    opacity_reg = _finite_float(
        getattr(opt, "mcmc_opacity_reg", 0.01), "mcmc_opacity_reg"
    )
    scale_reg = _finite_float(
        getattr(opt, "mcmc_scale_reg", 0.01), "mcmc_scale_reg"
    )
    growth_rate = _finite_float(
        getattr(opt, "mcmc_growth_rate", 1.05), "mcmc_growth_rate"
    )
    min_opacity = _finite_float(
        getattr(opt, "mcmc_min_opacity", 0.005), "mcmc_min_opacity"
    )
    if noise_lr < 0.0 or opacity_reg < 0.0 or scale_reg < 0.0:
        raise ValueError("MCMC noise and regularization coefficients must be non-negative")
    if growth_rate <= 1.0:
        raise ValueError("mcmc_growth_rate must be greater than 1")
    if not 0.0 < min_opacity < 1.0:
        raise ValueError("mcmc_min_opacity must be in (0, 1)")

    chunk_size = getattr(opt, "mcmc_noise_chunk_size", 250_000)
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
        raise TypeError("mcmc_noise_chunk_size must be an integer")
    if chunk_size <= 0:
        raise ValueError("mcmc_noise_chunk_size must be positive")

    densify_from = int(getattr(opt, "densify_from_iter"))
    densify_until = int(getattr(opt, "densify_until_iter"))
    interval = int(getattr(opt, "densification_interval"))
    if densify_from < 0 or densify_until <= densify_from:
        raise ValueError("MCMC densification window must be positive and ordered")
    if interval <= 0:
        raise ValueError("densification_interval must be positive")
    if str(getattr(opt, "optimizer_type", "default")).lower() != "default":
        raise ValueError("faithful MCMC requires --optimizer_type default")


def validate_mcmc_initialization(dataset) -> None:
    """Validate the independent random/SfM initialization contract."""
    init_type = str(getattr(dataset, "mcmc_init_type", "random")).lower()
    if init_type not in ("random", "sfm"):
        raise ValueError("mcmc_init_type must be 'random' or 'sfm'")
    random_points = getattr(dataset, "mcmc_random_points", 100_000)
    if isinstance(random_points, bool) or not isinstance(random_points, int):
        raise TypeError("mcmc_random_points must be an integer")
    if random_points <= 0:
        raise ValueError("mcmc_random_points must be positive")


def compute_mcmc_dataset_fingerprint(
    source_path,
    images="images",
    depths="",
    *,
    chunk_size=8 * 1024 * 1024,
) -> dict:
    """Hash every training image plus the COLMAP/depth inputs used on resume."""
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    root = Path(source_path).resolve()
    if not root.is_dir():
        raise ValueError("MCMC source_path does not exist: {}".format(root))

    files = []

    def add_tree(label, directory, required=False, suffixes=None):
        directory = Path(directory)
        if not directory.is_dir():
            if required:
                raise ValueError("Missing MCMC dataset directory: {}".format(directory))
            return
        selected = []
        for path in directory.rglob("*"):
            if not path.is_file():
                continue
            if suffixes is not None and path.suffix.lower() not in suffixes:
                continue
            logical_path = Path(label) / path.relative_to(directory)
            selected.append((logical_path.as_posix(), path))
        if required and not selected:
            raise ValueError("MCMC dataset directory is empty: {}".format(directory))
        files.extend(selected)

    image_dir = Path(images)
    if not image_dir.is_absolute():
        image_dir = root / image_dir
    add_tree("images", image_dir, required=True)

    if str(depths):
        depth_dir = Path(depths)
        if not depth_dir.is_absolute():
            depth_dir = root / depth_dir
        add_tree("depths", depth_dir, required=True)

    add_tree(
        "sparse/0",
        root / "sparse" / "0",
        required=True,
        suffixes={".bin", ".txt", ".json"},
    )

    digest = hashlib.sha256()
    total_bytes = 0
    for logical_path, path in sorted(files, key=lambda item: item[0]):
        size = path.stat().st_size
        total_bytes += size
        digest.update(logical_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        with open(path, "rb") as handle:
            while True:
                block = handle.read(chunk_size)
                if not block:
                    break
                digest.update(block)
        digest.update(b"\0")
    return {
        "algorithm": "sha256-tree-v1",
        "sha256": digest.hexdigest(),
        "file_count": len(files),
        "total_bytes": total_bytes,
    }


def compute_mcmc_growth_target(
    current_count: int, cap_max: int, growth_rate: float = 1.05
) -> int:
    """Return the next hard-capped Gaussian count used by 3DGS-MCMC."""
    values = (current_count, cap_max)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise TypeError("current_count and cap_max must be integers")
    if current_count < 0 or cap_max <= 0:
        raise ValueError("current_count must be non-negative and cap_max positive")
    growth_rate = _finite_float(growth_rate, "growth_rate")
    if growth_rate <= 1.0:
        raise ValueError("growth_rate must be greater than 1")
    if current_count >= cap_max:
        return cap_max
    return min(cap_max, int(growth_rate * current_count))


def build_binomial_coefficients(
    n_max: int = MCMC_RELOCATION_N_MAX,
    *,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build the dense binomial table expected by the relocation CUDA kernel."""
    if isinstance(n_max, bool) or not isinstance(n_max, int):
        raise TypeError("n_max must be an integer")
    if n_max < 2:
        raise ValueError("n_max must be at least 2")
    table = torch.zeros((n_max, n_max), dtype=torch.float64)
    for n in range(n_max):
        for k in range(n + 1):
            table[n, k] = math.comb(n, k)
    return table.to(device=device, dtype=dtype).contiguous()


def _validate_relocation_inputs(opacity_old, scale_old, multiplicity, n_max):
    opacity = torch.as_tensor(opacity_old)
    scale = torch.as_tensor(scale_old, device=opacity.device)
    multiplicity = torch.as_tensor(multiplicity, device=opacity.device)
    if opacity.ndim == 2 and opacity.shape[1] == 1:
        opacity = opacity[:, 0]
    if opacity.ndim != 1:
        raise ValueError("opacity_old must have shape [N] or [N, 1]")
    if scale.ndim != 2 or scale.shape != (opacity.shape[0], 3):
        raise ValueError("scale_old must have shape [N, 3]")
    multiplicity = multiplicity.reshape(-1)
    if multiplicity.shape[0] != opacity.shape[0]:
        raise ValueError("multiplicity must contain one value per Gaussian")
    if not torch.isfinite(opacity).all() or not torch.isfinite(scale).all():
        raise ValueError("relocation inputs must be finite")
    if torch.any((opacity < 0.0) | (opacity > 1.0)):
        raise ValueError("opacity_old must be in [0, 1]")
    multiplicity = multiplicity.to(dtype=torch.int64).clamp(1, n_max - 1)
    return opacity, scale, multiplicity


def compute_relocation_reference(
    opacity_old,
    scale_old,
    multiplicity,
    *,
    n_max: int = MCMC_RELOCATION_N_MAX,
):
    """Torch reference for Equation 9 and the official CUDA implementation.

    Computation uses float64 internally to make this suitable as a test oracle;
    outputs are converted back to the scale tensor's floating-point dtype.
    """
    if isinstance(n_max, bool) or not isinstance(n_max, int) or n_max < 2:
        raise ValueError("n_max must be an integer of at least 2")
    opacity, scale, multiplicity = _validate_relocation_inputs(
        opacity_old, scale_old, multiplicity, n_max
    )
    output_dtype = scale.dtype if scale.is_floating_point() else torch.float32
    opacity64 = opacity.to(dtype=torch.float64)
    scale64 = scale.to(dtype=torch.float64)
    multiplicity = multiplicity.to(device=opacity.device)
    new_opacity = 1.0 - torch.pow(1.0 - opacity64, 1.0 / multiplicity.to(torch.float64))
    binomials = build_binomial_coefficients(
        n_max, device=opacity.device, dtype=torch.float64
    )

    denominator = torch.zeros_like(new_opacity)
    max_multiplicity = int(multiplicity.max().item()) if multiplicity.numel() else 0
    for i in range(1, max_multiplicity + 1):
        active = multiplicity >= i
        if not bool(active.any()):
            continue
        terms = torch.zeros_like(new_opacity)
        for k in range(i):
            coefficient = ((-1.0) ** k) / math.sqrt(k + 1.0)
            terms += binomials[i - 1, k] * coefficient * new_opacity.pow(k + 1)
        denominator[active] += terms[active]

    if torch.any(~torch.isfinite(denominator)) or torch.any(denominator <= 0.0):
        raise RuntimeError("relocation denominator is non-positive or non-finite")
    new_scale = scale64 * (opacity64 / denominator).unsqueeze(-1)
    return new_opacity.to(dtype=output_dtype), new_scale.to(dtype=output_dtype)


def compute_relocation_cuda(
    opacity_old,
    scale_old,
    multiplicity,
    *,
    n_max: int = MCMC_RELOCATION_N_MAX,
):
    """Invoke the patched rasterizer relocation primitive lazily."""
    try:
        from diff_gaussian_rasterization import compute_relocation
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            "MCMC relocation requires the patched diff-gaussian-rasterization extension"
        ) from error

    opacity, scale, multiplicity = _validate_relocation_inputs(
        opacity_old, scale_old, multiplicity, n_max
    )
    binomials = build_binomial_coefficients(
        n_max, device=opacity.device, dtype=torch.float32
    )
    return compute_relocation(
        opacity.contiguous(),
        scale.contiguous(),
        multiplicity.to(dtype=torch.int32).contiguous(),
        binomials,
        n_max,
    )


def mcmc_noise_gate(
    opacity: torch.Tensor,
    min_opacity: float = 0.005,
    steepness: float = 100.0,
) -> torch.Tensor:
    """Return the official low-opacity SGLD gate with the corrected sign."""
    min_opacity = _finite_float(min_opacity, "min_opacity")
    steepness = _finite_float(steepness, "steepness")
    if not 0.0 < min_opacity < 1.0:
        raise ValueError("min_opacity must be in (0, 1)")
    if steepness <= 0.0:
        raise ValueError("steepness must be positive")
    return torch.sigmoid(steepness * (min_opacity - opacity))


def build_mcmc_resume_config(dataset, opt, pipe, seed, cap_max) -> dict:
    """Build the complete method configuration required for exact resume."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if isinstance(cap_max, bool) or not isinstance(cap_max, int) or cap_max <= 0:
        raise ValueError("cap_max must be a positive integer")
    validate_mcmc_initialization(dataset)
    dataset_fingerprint = compute_mcmc_dataset_fingerprint(
        dataset.source_path,
        images=getattr(dataset, "images", "images"),
        depths=getattr(dataset, "depths", ""),
    )
    cuda_available = torch.cuda.is_available()
    environment = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(0) if cuda_available else None,
        "cuda_capability": (
            list(torch.cuda.get_device_capability(0)) if cuda_available else None
        ),
    }
    return {
        "density_control": "mcmc",
        "optimization": dict(vars(opt)),
        "pipeline": {
            "antialiasing": bool(pipe.antialiasing),
            "compute_cov3D_python": bool(pipe.compute_cov3D_python),
            "convert_SHs_python": bool(pipe.convert_SHs_python),
        },
        "dataset": {
            "data_device": str(getattr(dataset, "data_device", "cuda")),
            "depths": str(getattr(dataset, "depths", "")),
            "eval": bool(getattr(dataset, "eval", False)),
            "images": str(getattr(dataset, "images", "images")),
            "mcmc_init_type": str(dataset.mcmc_init_type).lower(),
            "mcmc_random_points": int(dataset.mcmc_random_points),
            "resolution": int(getattr(dataset, "resolution", -1)),
            "sh_degree": int(dataset.sh_degree),
            "source_path": str(dataset.source_path),
            "train_test_exp": bool(dataset.train_test_exp),
            "white_background": bool(dataset.white_background),
            "fingerprint": dataset_fingerprint,
        },
        "environment": environment,
        "seed": int(seed),
        "cap_max": int(cap_max),
    }


def validate_mcmc_resume_config(runtime_state, current_config) -> None:
    """Reject a checkpoint made by another method or MCMC configuration."""
    if runtime_state.get("density_control") != "mcmc":
        raise ValueError("MCMC resume requires a checkpoint with density_control=mcmc")
    saved_config = runtime_state.get("resume_config")
    if saved_config is None:
        raise ValueError("MCMC checkpoint is missing its resume_config")
    if saved_config != current_config:
        differing_sections = [
            name
            for name in sorted(set(saved_config) | set(current_config))
            if saved_config.get(name) != current_config.get(name)
        ]
        raise ValueError(
            "MCMC checkpoint configuration mismatch in: {}".format(
                ", ".join(differing_sections)
            )
        )


def capture_mcmc_runtime_state(
    gaussians,
    viewpoint_indices,
    remaining_camera_names=None,
    camera_order_names=None,
    include_gradients=True,
) -> dict:
    """Capture RNG, camera-stack, exposure, and pending-gradient state."""
    parameter_grads = {}
    for group in gaussians.optimizer.param_groups:
        if len(group["params"]) != 1:
            raise RuntimeError("Each Gaussian optimizer group must own one tensor")
        gradient = group["params"][0].grad if include_gradients else None
        parameter_grads[group["name"]] = (
            None if gradient is None else gradient.detach().cpu().clone()
        )

    exposure_grads = []
    for group in gaussians.exposure_optimizer.param_groups:
        for parameter in group["params"]:
            gradient = parameter.grad if include_gradients else None
            exposure_grads.append(
                None if gradient is None else gradient.detach().cpu().clone()
            )

    return {
        "version": 1,
        "density_control": "mcmc",
        "viewpoint_indices": [int(index) for index in viewpoint_indices],
        "remaining_camera_names": (
            None
            if remaining_camera_names is None
            else [str(name) for name in remaining_camera_names]
        ),
        "camera_order_names": (
            None
            if camera_order_names is None
            else [str(name) for name in camera_order_names]
        ),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
        "parameter_grads": parameter_grads,
        "exposure": gaussians._exposure.detach().cpu().clone(),
        "exposure_mapping": dict(getattr(gaussians, "exposure_mapping", {})),
        "exposure_optimizer": gaussians.exposure_optimizer.state_dict(),
        "exposure_grads": exposure_grads,
    }


def restore_mcmc_runtime_state(gaussians, runtime_state) -> None:
    """Restore a state captured by :func:`capture_mcmc_runtime_state`."""
    if int(runtime_state.get("version", -1)) != 1:
        raise ValueError("Unsupported MCMC checkpoint runtime-state version")
    if runtime_state.get("density_control") != "mcmc":
        raise ValueError("Runtime state does not belong to MCMC")

    exposure = runtime_state["exposure"].to(
        device=gaussians._exposure.device, dtype=gaussians._exposure.dtype
    )
    if exposure.shape != gaussians._exposure.shape:
        raise ValueError("Checkpoint exposure tensor does not match this scene")
    with torch.no_grad():
        gaussians._exposure.copy_(exposure)
    exposure_mapping = runtime_state.get("exposure_mapping")
    if exposure_mapping:
        mapping_values = [int(index) for index in exposure_mapping.values()]
        if (
            len(exposure_mapping) != exposure.shape[0]
            or len(set(exposure_mapping)) != len(exposure_mapping)
            or sorted(mapping_values) != list(range(exposure.shape[0]))
        ):
            raise ValueError("Checkpoint exposure mapping is invalid")
        gaussians.exposure_mapping = {
            str(name): int(index) for name, index in exposure_mapping.items()
        }
    gaussians.exposure_optimizer.load_state_dict(runtime_state["exposure_optimizer"])

    parameter_grads = runtime_state.get("parameter_grads", {})
    expected_names = {group["name"] for group in gaussians.optimizer.param_groups}
    if set(parameter_grads) != expected_names:
        raise ValueError("Checkpoint Gaussian-gradient groups do not match the model")
    for group in gaussians.optimizer.param_groups:
        parameter = group["params"][0]
        saved_gradient = parameter_grads[group["name"]]
        parameter.grad = (
            None
            if saved_gradient is None
            else saved_gradient.to(device=parameter.device, dtype=parameter.dtype)
        )

    exposure_parameters = [
        parameter
        for group in gaussians.exposure_optimizer.param_groups
        for parameter in group["params"]
    ]
    saved_exposure_grads = runtime_state.get("exposure_grads", [])
    if len(saved_exposure_grads) != len(exposure_parameters):
        raise ValueError("Checkpoint exposure-gradient groups do not match the model")
    for parameter, saved_gradient in zip(exposure_parameters, saved_exposure_grads):
        parameter.grad = (
            None
            if saved_gradient is None
            else saved_gradient.to(device=parameter.device, dtype=parameter.dtype)
        )

    random.setstate(runtime_state["python_rng_state"])
    np.random.set_state(runtime_state["numpy_rng_state"])
    torch.set_rng_state(runtime_state["torch_rng_state"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(runtime_state["cuda_rng_state_all"])


def write_mcmc_config(dataset, config) -> str:
    """Persist ``mcmc_config.json`` beside the model's legacy ``cfg_args``."""
    model_path = str(dataset.model_path)
    os.makedirs(model_path, exist_ok=True)
    path = os.path.join(model_path, "mcmc_config.json")
    temporary_path = path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, indent=2, sort_keys=True)
        config_file.flush()
        os.fsync(config_file.fileno())
    os.replace(temporary_path, path)
    return path


__all__ = [
    "MCMC_RELOCATION_N_MAX",
    "build_binomial_coefficients",
    "build_mcmc_resume_config",
    "capture_mcmc_runtime_state",
    "compute_mcmc_dataset_fingerprint",
    "compute_mcmc_growth_target",
    "compute_relocation_cuda",
    "compute_relocation_reference",
    "mcmc_noise_gate",
    "restore_mcmc_runtime_state",
    "validate_mcmc_initialization",
    "validate_mcmc_options",
    "validate_mcmc_resume_config",
    "write_mcmc_config",
]
