import unittest
from argparse import ArgumentParser

from arguments import ModelParams, OptimizationParams


class McmcCliTests(unittest.TestCase):
    def setUp(self):
        self.parser = ArgumentParser()
        self.model = ModelParams(self.parser)
        self.optimization = OptimizationParams(self.parser)

    def test_mcmc_paper_defaults(self):
        args = self.parser.parse_args(["--density_control", "mcmc"])
        opt = self.optimization.extract(args)
        self.assertEqual(opt.density_control, "mcmc")
        self.assertEqual(opt.mcmc_init_mode, "paper")
        self.assertEqual(opt.mcmc_noise_lr, 500_000.0)
        self.assertEqual(opt.mcmc_opacity_reg, 0.01)
        self.assertEqual(opt.mcmc_scale_reg, 0.01)
        self.assertEqual(opt.mcmc_growth_rate, 1.05)
        self.assertEqual(opt.mcmc_min_opacity, 0.005)
        self.assertEqual(opt.mcmc_noise_chunk_size, 250_000)
        dataset = self.model.extract(args)
        self.assertEqual(dataset.mcmc_init_type, "random")
        self.assertEqual(dataset.mcmc_random_points, 100_000)

    def test_mcmc_ablation_values_parse(self):
        args = self.parser.parse_args(
            [
                "--density_control",
                "mcmc",
                "--mcmc_init_mode",
                "legacy",
                "--mcmc_noise_lr",
                "250000",
                "--mcmc_opacity_reg",
                "0.001",
                "--mcmc_scale_reg",
                "0.02",
                "--mcmc_growth_rate",
                "1.02",
                "--mcmc_min_opacity",
                "0.01",
                "--mcmc_noise_chunk_size",
                "100000",
                "--mcmc_init_type",
                "sfm",
                "--mcmc_random_points",
                "250000",
            ]
        )
        opt = self.optimization.extract(args)
        self.assertEqual(opt.mcmc_init_mode, "legacy")
        self.assertEqual(opt.mcmc_noise_lr, 250_000.0)
        self.assertEqual(opt.mcmc_opacity_reg, 0.001)
        self.assertEqual(opt.mcmc_scale_reg, 0.02)
        self.assertEqual(opt.mcmc_growth_rate, 1.02)
        self.assertEqual(opt.mcmc_min_opacity, 0.01)
        self.assertEqual(opt.mcmc_noise_chunk_size, 100_000)
        dataset = self.model.extract(args)
        self.assertEqual(dataset.mcmc_init_type, "sfm")
        self.assertEqual(dataset.mcmc_random_points, 250_000)

    def test_default_3dgs_path_remains_unchanged(self):
        opt = self.optimization.extract(self.parser.parse_args([]))
        self.assertEqual(opt.density_control, "3dgs")


if __name__ == "__main__":
    unittest.main()
