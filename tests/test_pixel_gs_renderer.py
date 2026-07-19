import importlib.util
import sys
import types
import unittest
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]


RasterizationSettings = namedtuple(
    "GaussianRasterizationSettings",
    (
        "image_height",
        "image_width",
        "tanfovx",
        "tanfovy",
        "bg",
        "scale_modifier",
        "viewmatrix",
        "projmatrix",
        "sh_degree",
        "campos",
        "prefiltered",
        "debug",
        "antialiasing",
        "pixel_weights",
        "track_pixel_counts",
    ),
)


class FakeRasterizer:
    def __init__(self, raster_settings):
        self.settings = raster_settings

    def __call__(self, **kwargs):
        count = kwargs["means3D"].shape[0]
        image = torch.zeros((3, 2, 2), dtype=torch.float32)
        radii = torch.ones(count, dtype=torch.float32)
        depth = torch.zeros((1, 2, 2), dtype=torch.float32)
        outputs = [image, radii, depth]
        if self.settings.pixel_weights is not None:
            outputs.append(torch.ones(count, dtype=torch.float32))
        if self.settings.track_pixel_counts:
            outputs.append(torch.arange(1, count + 1, dtype=torch.float32))
        return tuple(outputs)


def _load_renderer_module():
    fake_rasterizer = types.ModuleType("diff_gaussian_rasterization")
    fake_rasterizer.GaussianRasterizationSettings = RasterizationSettings
    fake_rasterizer.GaussianRasterizer = FakeRasterizer

    fake_model = types.ModuleType("scene.gaussian_model")
    fake_model.GaussianModel = object

    replacements = {
        "diff_gaussian_rasterization": fake_rasterizer,
        "scene.gaussian_model": fake_model,
    }
    previous = {name: sys.modules.get(name) for name in replacements}
    sys.modules.update(replacements)
    try:
        spec = importlib.util.spec_from_file_location(
            "pixelgs_renderer_under_test", ROOT / "gaussian_renderer" / "__init__.py"
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        for name, old_module in previous.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module


RENDERER = _load_renderer_module()


class DummyGaussians:
    def __init__(self):
        self.xyz = torch.tensor(
            [[0.0, 0.0, 0.5], [0.0, 0.0, 2.0]], dtype=torch.float32
        )
        self.active_sh_degree = 0
        self.max_sh_degree = 0

    @property
    def get_xyz(self):
        return self.xyz

    @property
    def get_opacity(self):
        return torch.ones((2, 1), dtype=torch.float32)

    @property
    def get_scaling(self):
        return torch.ones((2, 3), dtype=torch.float32)

    @property
    def get_rotation(self):
        return torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1)

    @property
    def get_features(self):
        return torch.zeros((2, 1, 3), dtype=torch.float32)


class PixelGsRendererTests(unittest.TestCase):
    def test_depth_scale_and_pixel_count_output(self):
        camera = SimpleNamespace(
            FoVx=1.0,
            FoVy=1.0,
            image_height=2,
            image_width=2,
            world_view_transform=torch.eye(4),
            full_proj_transform=torch.eye(4),
            camera_center=torch.zeros(3),
        )
        pipe = SimpleNamespace(
            debug=False,
            antialiasing=False,
            compute_cov3D_python=False,
            convert_SHs_python=False,
        )

        result = RENDERER.render(
            camera,
            DummyGaussians(),
            pipe,
            torch.zeros(3),
            track_pixel_counts=True,
            pixelgs_depth_threshold=1.0,
        )

        self.assertTrue(
            torch.equal(result["pixelgs_grad_scale"], torch.tensor([0.25, 1.0]))
        )
        self.assertTrue(
            torch.equal(result["pixel_counts"], torch.tensor([1.0, 2.0]))
        )
        self.assertEqual(result["visibility_filter"].flatten().tolist(), [0, 1])


if __name__ == "__main__":
    unittest.main()
