import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch

from utils.mcmc_utils import compute_relocation_reference


ROOT = Path(__file__).resolve().parents[1]


def _load_gaussian_model_module():
    if "plyfile" not in sys.modules:
        plyfile = types.ModuleType("plyfile")
        plyfile.PlyData = object
        plyfile.PlyElement = object
        sys.modules["plyfile"] = plyfile
    if "simple_knn._C" not in sys.modules:
        simple_knn = types.ModuleType("simple_knn")
        simple_knn.__path__ = []
        simple_knn_c = types.ModuleType("simple_knn._C")
        simple_knn_c.distCUDA2 = lambda tensor: torch.ones(
            tensor.shape[0], dtype=tensor.dtype, device=tensor.device
        )
        sys.modules["simple_knn"] = simple_knn
        sys.modules["simple_knn._C"] = simple_knn_c
    spec = importlib.util.spec_from_file_location(
        "mcmc_gaussian_model_under_test", ROOT / "scene" / "gaussian_model.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


GM = _load_gaussian_model_module()


def _make_model(count=20, opacities=None, init_mode="legacy", populate_state=False):
    if opacities is None:
        opacities = torch.full((count, 1), 0.8, dtype=torch.float32)
    else:
        opacities = torch.as_tensor(opacities, dtype=torch.float32).reshape(count, 1)
    model = GM.GaussianModel(sh_degree=0, mcmc_init_mode=init_mode)
    model._xyz = torch.nn.Parameter(
        torch.stack(
            (
                torch.arange(count, dtype=torch.float32),
                torch.zeros(count),
                torch.zeros(count),
            ),
            dim=1,
        )
    )
    model._features_dc = torch.nn.Parameter(torch.zeros((count, 1, 3)))
    model._features_rest = torch.nn.Parameter(torch.zeros((count, 0, 3)))
    model._scaling = torch.nn.Parameter(torch.zeros((count, 3)))
    model._rotation = torch.nn.Parameter(
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(count, 1)
    )
    model._opacity = torch.nn.Parameter(GM.inverse_sigmoid(opacities))
    groups = [
        {"params": [model._xyz], "name": "xyz"},
        {"params": [model._features_dc], "name": "f_dc"},
        {"params": [model._features_rest], "name": "f_rest"},
        {"params": [model._opacity], "name": "opacity"},
        {"params": [model._scaling], "name": "scaling"},
        {"params": [model._rotation], "name": "rotation"},
    ]
    model.optimizer = torch.optim.Adam(groups, lr=0.01)
    model.max_radii2D = torch.zeros(count)
    model.xyz_gradient_accum = torch.zeros((count, 1))
    model.xyz_gradient_accum_abs = None
    model.denom = torch.zeros((count, 1))
    model.tmp_radii = None
    if populate_state:
        for group in model.optimizer.param_groups:
            group["params"][0].grad = torch.ones_like(group["params"][0])
        model.optimizer.step()
        model.optimizer.zero_grad(set_to_none=True)
    return model


class InitializationTests(unittest.TestCase):
    def test_paper_and_legacy_initialization_are_isolated(self):
        dist2 = torch.tensor([1.0, 4.0])
        paper = GM.GaussianModel(0, mcmc_init_mode="paper")
        legacy = GM.GaussianModel(0, mcmc_init_mode="legacy")
        paper_scale, paper_opacity = paper._initial_scale_and_opacity(dist2)
        legacy_scale, legacy_opacity = legacy._initial_scale_and_opacity(dist2)
        self.assertTrue(
            torch.allclose(torch.exp(paper_scale), torch.exp(legacy_scale) * 0.1)
        )
        self.assertTrue(
            torch.allclose(torch.sigmoid(paper_opacity), torch.full((2, 1), 0.5))
        )
        self.assertTrue(
            torch.allclose(torch.sigmoid(legacy_opacity), torch.full((2, 1), 0.1))
        )

    def test_unknown_initialization_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "mcmc_init_mode"):
            GM.GaussianModel(0, mcmc_init_mode="random")


class RelocationAndGrowthTests(unittest.TestCase):
    def test_dead_gaussian_is_relocated_and_adam_moments_are_cleared(self):
        model = _make_model(
            count=2, opacities=[0.8, 0.001], populate_state=True
        )
        old_parameters = [group["params"][0] for group in model.optimizer.param_groups]
        for parameter in old_parameters:
            parameter.grad = torch.ones_like(parameter)
        relocated = model.relocate_mcmc(
            torch.tensor([False, True]),
            relocation_fn=compute_relocation_reference,
        )
        self.assertEqual(relocated, 1)
        self.assertTrue(torch.allclose(model.get_xyz[0], model.get_xyz[1]))
        self.assertTrue(torch.allclose(model.get_opacity[0], model.get_opacity[1]))
        self.assertTrue(torch.isfinite(model.get_scaling).all())
        for old_parameter, group in zip(old_parameters, model.optimizer.param_groups):
            self.assertIsNot(old_parameter, group["params"][0])
            self.assertIsNone(group["params"][0].grad)
            state = model.optimizer.state[group["params"][0]]
            self.assertEqual(int(torch.count_nonzero(state["exp_avg"][0])), 0)
            self.assertEqual(int(torch.count_nonzero(state["exp_avg_sq"][0])), 0)
            if state["exp_avg"][1].numel() > 0:
                self.assertGreater(int(torch.count_nonzero(state["exp_avg"][1])), 0)

    def test_growth_is_five_percent_and_never_exceeds_cap(self):
        model = _make_model(count=20)
        generator = torch.Generator().manual_seed(3)
        added = model.grow_mcmc(
            cap_max=21,
            growth_rate=1.05,
            relocation_fn=compute_relocation_reference,
            generator=generator,
        )
        self.assertEqual(added, 1)
        self.assertEqual(model.get_xyz.shape[0], 21)
        self.assertEqual(model.xyz_gradient_accum.shape[0], 21)
        self.assertEqual(model.denom.shape[0], 21)
        self.assertEqual(model.max_radii2D.shape[0], 21)
        self.assertEqual(
            model.grow_mcmc(
                cap_max=21,
                relocation_fn=compute_relocation_reference,
                generator=generator,
            ),
            0,
        )

    def test_coupled_event_reports_relocation_and_growth(self):
        opacity = [0.001] + [0.8] * 19
        model = _make_model(count=20, opacities=opacity)
        report = model.mcmc_relocate_and_grow(
            cap_max=21,
            relocation_fn=compute_relocation_reference,
            generator=torch.Generator().manual_seed(9),
        )
        self.assertEqual(
            report,
            {
                "before": 20,
                "relocated": 1,
                "added": 1,
                "after": 21,
                "cap_max": 21,
            },
        )
        self.assertTrue(torch.isfinite(model.get_xyz).all())
        self.assertTrue(torch.isfinite(model.get_opacity).all())
        self.assertTrue(torch.isfinite(model.get_scaling).all())

    def test_saturated_donor_opacity_is_clamped_before_relocation(self):
        model = _make_model(count=2, opacities=[0.8, 0.001])
        with torch.no_grad():
            model._opacity[0] = 100.0

        def checked_relocation(opacity, scale, multiplicity, **kwargs):
            self.assertTrue(torch.all(opacity < 1.0))
            return compute_relocation_reference(
                opacity, scale, multiplicity, **kwargs
            )

        relocated = model.relocate_mcmc(
            torch.tensor([False, True]), relocation_fn=checked_relocation
        )
        self.assertEqual(relocated, 1)
        self.assertTrue(torch.isfinite(model.get_scaling).all())


class NoiseTests(unittest.TestCase):
    def test_chunked_noise_is_deterministic_and_moves_low_opacity_points(self):
        low_opacity = [0.001] * 5
        first = _make_model(count=5, opacities=low_opacity)
        second = _make_model(count=5, opacities=low_opacity)
        original = first.get_xyz.detach().clone()
        processed = first.add_mcmc_position_noise(
            noise_lr=2.0,
            xyz_lr=0.1,
            chunk_size=2,
            generator=torch.Generator().manual_seed(123),
        )
        second.add_mcmc_position_noise(
            noise_lr=2.0,
            xyz_lr=0.1,
            chunk_size=2,
            generator=torch.Generator().manual_seed(123),
        )
        self.assertEqual(processed, 5)
        self.assertFalse(torch.equal(first.get_xyz, original))
        self.assertTrue(torch.equal(first.get_xyz, second.get_xyz))

    def test_zero_noise_is_a_no_op(self):
        model = _make_model(count=3, opacities=[0.001] * 3)
        original = model.get_xyz.detach().clone()
        self.assertEqual(model.add_mcmc_position_noise(0.0, 0.1, chunk_size=1), 3)
        self.assertTrue(torch.equal(model.get_xyz, original))


if __name__ == "__main__":
    unittest.main()
