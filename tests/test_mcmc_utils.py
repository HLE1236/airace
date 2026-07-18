import json
import copy
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from utils.mcmc_utils import (
    build_binomial_coefficients,
    build_mcmc_resume_config,
    capture_mcmc_runtime_state,
    compute_mcmc_dataset_fingerprint,
    compute_mcmc_growth_target,
    compute_relocation_reference,
    mcmc_noise_gate,
    restore_mcmc_runtime_state,
    validate_mcmc_initialization,
    validate_mcmc_options,
    validate_mcmc_resume_config,
    write_mcmc_config,
)


def _dataset_tree(root: Path) -> SimpleNamespace:
    image_dir = root / "images"
    sparse_dir = root / "sparse" / "0"
    image_dir.mkdir(parents=True)
    sparse_dir.mkdir(parents=True)
    (image_dir / "0001.jpg").write_bytes(b"image-one")
    (image_dir / "0002.jpg").write_bytes(b"image-two")
    (sparse_dir / "cameras.bin").write_bytes(b"cameras")
    (sparse_dir / "images.bin").write_bytes(b"poses")
    (sparse_dir / "points3D.bin").write_bytes(b"points")
    return SimpleNamespace(
        data_device="cuda",
        depths="",
        eval=False,
        images="images",
        mcmc_init_type="random",
        mcmc_random_points=100_000,
        resolution=1,
        sh_degree=3,
        source_path=str(root),
        train_test_exp=False,
        white_background=False,
    )


def _options(**overrides):
    values = dict(
        density_control="mcmc",
        optimizer_type="default",
        densify_from_iter=500,
        densify_until_iter=25_000,
        densification_interval=100,
        mcmc_init_mode="paper",
        mcmc_noise_lr=500_000.0,
        mcmc_opacity_reg=0.01,
        mcmc_scale_reg=0.01,
        mcmc_growth_rate=1.05,
        mcmc_min_opacity=0.005,
        mcmc_noise_chunk_size=250_000,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class ConfigurationTests(unittest.TestCase):
    def test_paper_configuration_is_valid(self):
        validate_mcmc_options(_options(), cap_max=5_100_000)

    def test_invalid_coupled_options_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "cap_max"):
            validate_mcmc_options(_options(), cap_max=0)
        with self.assertRaisesRegex(ValueError, "growth_rate"):
            validate_mcmc_options(_options(mcmc_growth_rate=1.0), cap_max=10)
        with self.assertRaisesRegex(ValueError, "init_mode"):
            validate_mcmc_options(_options(mcmc_init_mode="random"), cap_max=10)
        with self.assertRaisesRegex(ValueError, "optimizer_type"):
            validate_mcmc_options(_options(optimizer_type="sparse_adam"), cap_max=10)

    def test_growth_is_floor_and_hard_capped(self):
        self.assertEqual(compute_mcmc_growth_target(100, 1_000), 105)
        self.assertEqual(compute_mcmc_growth_target(100, 102), 102)
        self.assertEqual(compute_mcmc_growth_target(102, 102), 102)

    def test_random_initialization_contract_is_validated(self):
        validate_mcmc_initialization(
            SimpleNamespace(mcmc_init_type="random", mcmc_random_points=100_000)
        )
        with self.assertRaisesRegex(ValueError, "init_type"):
            validate_mcmc_initialization(
                SimpleNamespace(mcmc_init_type="unknown", mcmc_random_points=100)
            )
        with self.assertRaisesRegex(ValueError, "positive"):
            validate_mcmc_initialization(
                SimpleNamespace(mcmc_init_type="random", mcmc_random_points=0)
            )

    def test_dataset_fingerprint_tracks_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _dataset_tree(root)
            first = compute_mcmc_dataset_fingerprint(root)
            second = compute_mcmc_dataset_fingerprint(root)
            self.assertEqual(first, second)
            self.assertEqual(first["file_count"], 5)
            (root / "images" / "0002.jpg").write_bytes(b"changed-image")
            changed = compute_mcmc_dataset_fingerprint(root)
            self.assertNotEqual(first["sha256"], changed["sha256"])


class RelocationMathTests(unittest.TestCase):
    def test_binomial_table(self):
        table = build_binomial_coefficients(6)
        self.assertEqual(tuple(table.shape), (6, 6))
        self.assertEqual(float(table[5, 2]), 10.0)
        self.assertEqual(float(table[3, 4]), 0.0)

    def test_reference_n_one_is_identity(self):
        opacity = torch.tensor([0.2, 0.8])
        scale = torch.tensor([[1.0, 2.0, 3.0], [0.5, 0.75, 1.25]])
        multiplicity = torch.ones(2, dtype=torch.long)
        new_opacity, new_scale = compute_relocation_reference(
            opacity, scale, multiplicity
        )
        self.assertTrue(torch.allclose(new_opacity, opacity, atol=1e-6))
        self.assertTrue(torch.allclose(new_scale, scale, atol=1e-6))
        self.assertTrue(torch.equal(multiplicity, torch.ones_like(multiplicity)))

    def test_reference_multiple_copies_is_finite_and_reduces_opacity(self):
        opacity = torch.tensor([0.8])
        scale = torch.tensor([[1.0, 2.0, 3.0]])
        new_opacity, new_scale = compute_relocation_reference(
            opacity, scale, torch.tensor([3])
        )
        self.assertLess(float(new_opacity), float(opacity))
        self.assertTrue(torch.isfinite(new_scale).all())
        self.assertTrue(torch.all(new_scale > 0))

    def test_reference_accepts_float32_saturated_opacity(self):
        opacity = torch.ones(1, dtype=torch.float32)
        scale = torch.ones((1, 3), dtype=torch.float32)
        new_opacity, new_scale = compute_relocation_reference(
            opacity, scale, torch.tensor([3])
        )
        self.assertEqual(float(new_opacity), 1.0)
        self.assertTrue(torch.isfinite(new_scale).all())
        self.assertTrue(torch.all(new_scale > 0))

    def test_noise_gate_has_correct_sign(self):
        opacity = torch.tensor([0.0, 0.005, 0.5])
        gate = mcmc_noise_gate(opacity)
        self.assertGreater(float(gate[0]), float(gate[2]))
        self.assertAlmostEqual(float(gate[1]), 0.5, places=6)


class CheckpointTests(unittest.TestCase):
    class DummyGaussians:
        def __init__(self):
            self.parameter = torch.nn.Parameter(torch.tensor([[1.0, 2.0]]))
            self.optimizer = torch.optim.Adam(
                [{"params": [self.parameter], "name": "xyz"}], lr=0.01
            )
            self._exposure = torch.nn.Parameter(torch.tensor([[0.5, 0.25]]))
            self.exposure_mapping = {"camera_a": 0}
            self.exposure_optimizer = torch.optim.Adam([self._exposure], lr=0.01)

    def test_runtime_round_trip_restores_rng_exposure_and_gradients(self):
        gaussians = self.DummyGaussians()
        gaussians.parameter.grad = torch.tensor([[2.0, 3.0]])
        gaussians._exposure.grad = torch.tensor([[4.0, 5.0]])
        random.seed(77)
        np.random.seed(77)
        torch.manual_seed(77)
        runtime = capture_mcmc_runtime_state(
            gaussians,
            [3, 1],
            remaining_camera_names=["camera_d", "camera_b"],
            camera_order_names=["camera_a"],
        )
        expected_python = random.random()
        expected_numpy = float(np.random.rand())
        expected_torch = torch.rand(4)

        with torch.no_grad():
            gaussians._exposure.zero_()
        gaussians.parameter.grad = None
        gaussians._exposure.grad = None
        random.seed(1)
        np.random.seed(1)
        torch.manual_seed(1)
        restore_mcmc_runtime_state(gaussians, runtime)

        self.assertTrue(torch.equal(gaussians._exposure, torch.tensor([[0.5, 0.25]])))
        self.assertTrue(torch.equal(gaussians.parameter.grad, torch.tensor([[2.0, 3.0]])))
        self.assertEqual(random.random(), expected_python)
        self.assertEqual(float(np.random.rand()), expected_numpy)
        self.assertTrue(torch.equal(torch.rand(4), expected_torch))
        self.assertEqual(runtime["viewpoint_indices"], [3, 1])

    def test_final_runtime_state_omits_dense_gradients(self):
        gaussians = self.DummyGaussians()
        gaussians.parameter.grad = torch.ones_like(gaussians.parameter)
        gaussians._exposure.grad = torch.ones_like(gaussians._exposure)
        runtime = capture_mcmc_runtime_state(
            gaussians, [], include_gradients=False
        )
        self.assertEqual(runtime["parameter_grads"], {"xyz": None})
        self.assertEqual(runtime["exposure_grads"], [None])

    def test_resume_config_validation_and_json_write(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            dataset = _dataset_tree(source)
            pipe = SimpleNamespace(
                antialiasing=False,
                compute_cov3D_python=False,
                convert_SHs_python=False,
            )
            config = build_mcmc_resume_config(
                dataset, _options(), pipe, seed=0, cap_max=5_100_000
            )
            validate_mcmc_resume_config(
                {"density_control": "mcmc", "resume_config": config}, config
            )
            changed = copy.deepcopy(config)
            changed["cap_max"] = 4_500_000
            with self.assertRaisesRegex(ValueError, "cap_max"):
                validate_mcmc_resume_config(
                    {"density_control": "mcmc", "resume_config": changed}, config
                )
            changed = copy.deepcopy(config)
            changed["dataset"]["resolution"] = 2
            with self.assertRaisesRegex(ValueError, "dataset"):
                validate_mcmc_resume_config(
                    {"density_control": "mcmc", "resume_config": changed}, config
                )

            dataset.model_path = directory
            path = Path(write_mcmc_config(dataset, config))
            self.assertTrue(path.is_file())
            self.assertFalse(Path(str(path) + ".tmp").exists())
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), config)


if __name__ == "__main__":
    unittest.main()
