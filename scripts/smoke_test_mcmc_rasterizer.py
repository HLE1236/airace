"""CUDA smoke test for the analytical 3DGS-MCMC relocation kernel."""

from __future__ import annotations

import math

import torch


def _binomial_table(n_max: int, device: torch.device) -> torch.Tensor:
    table = torch.zeros((n_max, n_max), dtype=torch.float32, device=device)
    for n in range(n_max):
        for k in range(n + 1):
            table[n, k] = math.comb(n, k)
    return table


def _reference_relocation(
    opacity_old: torch.Tensor,
    scale_old: torch.Tensor,
    multiplicities: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Small float64 CPU reference for Equation (9)."""
    opacity = opacity_old.detach().to(device="cpu", dtype=torch.float64)
    scales = scale_old.detach().to(device="cpu", dtype=torch.float64)
    counts = multiplicities.detach().to(device="cpu", dtype=torch.int64)
    opacity_new = torch.empty_like(opacity)
    scale_new = torch.empty_like(scales)

    for index, count_tensor in enumerate(counts):
        count = int(count_tensor)
        updated_opacity = 1.0 - (1.0 - float(opacity[index])) ** (1.0 / count)
        denominator = 0.0
        for i in range(1, count + 1):
            for k in range(i):
                denominator += (
                    math.comb(i - 1, k)
                    * ((-1.0) ** k)
                    / math.sqrt(k + 1)
                    * (updated_opacity ** (k + 1))
                )
        opacity_new[index] = updated_opacity
        scale_new[index] = scales[index] * float(opacity[index]) / denominator

    return opacity_new, scale_new.reshape(-1)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This smoke test requires a CUDA-enabled PyTorch build")

    from diff_gaussian_rasterization import compute_relocation

    device = torch.device("cuda")
    n_max = 8
    opacity_old = torch.tensor(
        [0.2, 0.6, 0.005, 0.85], dtype=torch.float32, device=device
    )
    scale_old = torch.tensor(
        [
            [0.5, 0.25, 0.125],
            [1.0, 0.75, 0.5],
            [0.02, 0.03, 0.04],
            [2.0, 1.0, 0.5],
        ],
        dtype=torch.float32,
        device=device,
    )
    multiplicities = torch.tensor([1, 2, 3, 7], dtype=torch.int64, device=device)
    binoms = _binomial_table(n_max, device)

    opacity_new, scale_new = compute_relocation(
        opacity_old, scale_old, multiplicities, binoms, n_max
    )
    torch.cuda.synchronize()

    if opacity_new.shape != opacity_old.shape:
        raise AssertionError("Relocated opacity must have shape [P]")
    if scale_new.shape != (3 * opacity_old.shape[0],):
        raise AssertionError("Relocated scale must preserve the official flat [3P] ABI")
    if opacity_new.requires_grad or scale_new.requires_grad:
        raise AssertionError("Relocation is a non-differentiable structural operation")
    if not torch.isfinite(opacity_new).all() or not torch.isfinite(scale_new).all():
        raise AssertionError("Relocation produced non-finite values")
    if not torch.all(scale_new > 0):
        raise AssertionError("Relocation produced a non-positive scale")

    expected_opacity, expected_scale = _reference_relocation(
        opacity_old, scale_old, multiplicities
    )
    if not torch.allclose(
        opacity_new.cpu().double(), expected_opacity, rtol=2e-5, atol=2e-7
    ):
        raise AssertionError("CUDA relocation opacity does not match CPU reference")
    if not torch.allclose(
        scale_new.cpu().double(), expected_scale, rtol=3e-4, atol=2e-6
    ):
        raise AssertionError("CUDA relocation scale does not match CPU reference")

    reconstructed = 1.0 - torch.pow(
        1.0 - opacity_new, multiplicities.to(dtype=torch.float32)
    )
    if not torch.allclose(reconstructed, opacity_old, rtol=2e-5, atol=2e-7):
        raise AssertionError("Relocated opacity does not preserve composed alpha")

    empty_opacity, empty_scale = compute_relocation(
        opacity_old[:0], scale_old[:0], multiplicities[:0], binoms, n_max
    )
    if empty_opacity.numel() != 0 or empty_scale.numel() != 0:
        raise AssertionError("Empty relocation inputs must return empty outputs")

    print(
        "3DGS-MCMC relocation smoke test passed on {} (P={}, max_N={}).".format(
            torch.cuda.get_device_name(0), opacity_old.numel(), int(multiplicities.max())
        )
    )


if __name__ == "__main__":
    main()
