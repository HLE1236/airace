"""Tiny GPU forward/backward smoke test for the patched rasterizer."""

from __future__ import annotations

import torch

from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)


def _render(settings, means2d):
    device = means2d.device
    rasterizer = GaussianRasterizer(raster_settings=settings)
    return rasterizer(
        # The second Gaussian is deliberately outside the view frustum so
        # Pixel-GS coverage must remain zero for it.
        means3D=torch.tensor(
            [[0.0, 0.0, 2.0], [100.0, 0.0, 2.0], [0.65, 0.0, 2.0]],
            device=device,
        ),
        means2D=means2d,
        colors_precomp=torch.tensor(
            [[0.8, 0.4, 0.2], [0.2, 0.4, 0.8], [0.4, 0.8, 0.2]],
            device=device,
        ),
        opacities=torch.tensor([[0.8], [0.8], [0.8]], device=device),
        scales=torch.tensor(
            [
                [0.35, 0.25, 0.20],
                [0.35, 0.25, 0.20],
                [0.06, 0.06, 0.06],
            ],
            device=device,
        ),
        rotations=torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ],
            device=device,
        ),
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This smoke test requires a CUDA-enabled PyTorch build")

    device = torch.device("cuda")
    height = width = 32
    common = dict(
        image_height=height,
        image_width=width,
        tanfovx=1.0,
        tanfovy=1.0,
        bg=torch.zeros(3, device=device),
        scale_modifier=1.0,
        viewmatrix=torch.eye(4, device=device),
        projmatrix=torch.eye(4, device=device),
        sh_degree=0,
        campos=torch.zeros(3, device=device),
        prefiltered=False,
        debug=False,
        antialiasing=False,
    )

    # Legacy callers still use an (N,3) screen-space dummy and expect three
    # public outputs plus a zero third gradient component.
    legacy_means2d = torch.zeros((3, 3), device=device, requires_grad=True)
    baseline_outputs = _render(
        GaussianRasterizationSettings(
            **common, pixel_weights=None, track_pixel_counts=False
        ),
        legacy_means2d,
    )
    if len(baseline_outputs) != 3:
        raise AssertionError("Baseline rasterizer API must return three outputs")
    baseline_outputs[0].sum().backward()
    if legacy_means2d.grad is None or float(legacy_means2d.grad[0, 2]) != 0.0:
        raise AssertionError("Legacy means2D third gradient must remain zero")

    # EAS requests a fourth output and AbsGrad writes positive absolute
    # contributions into screen-gradient channels 2-3.
    means2d = torch.zeros((3, 4), device=device, requires_grad=True)
    pixel_weights = torch.zeros((height, width), device=device)
    pixel_weights[:, width // 2 :] = 1.0
    image, radii, depth, accum_weights = _render(
        GaussianRasterizationSettings(
            **common, pixel_weights=pixel_weights, track_pixel_counts=False
        ),
        means2d,
    )
    image.sum().backward()

    if (
        radii.numel() != 3
        or int(radii[0]) <= int(radii[2])
        or int(radii[2]) <= 0
        or int(radii[1]) != 0
    ):
        raise AssertionError("Smoke Gaussian visibility/radius setup is invalid")
    if accum_weights.shape != (3,) or not torch.isfinite(accum_weights).all():
        raise AssertionError("EAS must return one finite score per Gaussian")
    if float(accum_weights[0]) <= 0.0 or float(accum_weights[1]) != 0.0:
        raise AssertionError("EAS scores do not match visible/hidden Gaussians")
    if means2d.grad is None or not torch.isfinite(means2d.grad).all():
        raise AssertionError("AbsGrad screen gradients are missing or non-finite")
    if float(means2d.grad[0, 2:].sum()) <= 0.0:
        raise AssertionError("AbsGrad channels did not accumulate positive gradients")
    if not torch.isfinite(image).all() or not torch.isfinite(depth).all():
        raise AssertionError("Rasterizer produced non-finite image/depth values")

    # Pixel-GS without EAS uses the fourth public output slot.
    coverage_means2d = torch.zeros((3, 4), device=device, requires_grad=True)
    coverage_image, coverage_radii, coverage_depth, pixel_counts = _render(
        GaussianRasterizationSettings(
            **common, pixel_weights=None, track_pixel_counts=True
        ),
        coverage_means2d,
    )
    if pixel_counts.shape != (3,) or pixel_counts.requires_grad:
        raise AssertionError("Pixel counts must be a non-differentiable [N] tensor")
    if not torch.isfinite(pixel_counts).all():
        raise AssertionError("Pixel counts must be finite")
    if (
        float(pixel_counts[0]) <= float(pixel_counts[2])
        or float(pixel_counts[2]) <= 0.0
        or float(pixel_counts[1]) != 0.0
    ):
        raise AssertionError(
            "Pixel counts must rank large > small > hidden Gaussians"
        )
    coverage_image.sum().backward()
    if coverage_means2d.grad is None or not torch.isfinite(coverage_means2d.grad).all():
        raise AssertionError("Pixel counting must preserve differentiable rendering")

    # EAS and Pixel-GS can be enabled together; their two auxiliary outputs
    # remain separate and appear in a stable order.
    both_means2d = torch.zeros((3, 4), device=device, requires_grad=True)
    both_outputs = _render(
        GaussianRasterizationSettings(
            **common, pixel_weights=pixel_weights, track_pixel_counts=True
        ),
        both_means2d,
    )
    if len(both_outputs) != 5:
        raise AssertionError("Combined EAS/Pixel-GS API must return five outputs")
    both_image, both_radii, both_depth, both_accum_weights, both_pixel_counts = both_outputs
    if not torch.equal(pixel_counts, both_pixel_counts):
        raise AssertionError("EAS must not change Pixel-GS coverage counts")
    if not torch.allclose(accum_weights, both_accum_weights):
        raise AssertionError("Pixel counting must not change EAS accumulation")
    if not torch.equal(coverage_radii, both_radii):
        raise AssertionError("Auxiliary tracking must not change Gaussian radii")
    if not torch.allclose(coverage_image, both_image) or not torch.allclose(coverage_depth, both_depth):
        raise AssertionError("Auxiliary tracking must not change image/depth output")
    both_image.sum().backward()
    if both_means2d.grad is None or not torch.isfinite(both_means2d.grad).all():
        raise AssertionError("Combined tracking must preserve backward gradients")

    print(
        "Improved-GS/Pixel-GS rasterizer smoke test passed on {} "
        "(AbsGrad={}, EAS={:.6g}, covered_pixels={:.0f}).".format(
            torch.cuda.get_device_name(0),
            means2d.grad[0, 2:].detach().cpu().tolist(),
            float(accum_weights[0]),
            float(pixel_counts[0]),
        )
    )


if __name__ == "__main__":
    main()
