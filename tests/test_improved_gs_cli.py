import unittest
import ast
from argparse import ArgumentParser
from pathlib import Path

import torch

from arguments import OptimizationParams


ROOT = Path(__file__).resolve().parents[1]


def _load_density_control_validator():
    """Load the pure validation helper without importing train.py's CUDA stack."""
    source = (ROOT / "train.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_validate_density_control_options"
    )
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "train.py", "exec"), namespace)
    return namespace[function.name]


VALIDATE_DENSITY_CONTROL = _load_density_control_validator()


class ImprovedGsCliTests(unittest.TestCase):
    def test_baseline_remains_default(self):
        parser = ArgumentParser()
        group = OptimizationParams(parser)
        parsed = group.extract(parser.parse_args([]))
        self.assertEqual(parsed.density_control, "3dgs")
        self.assertEqual(parsed.gaussian_budget, 1_500_000)
        self.assertAlmostEqual(parsed.pixelgs_depth_threshold, 0.37)

    def test_component_ablation_switches_accept_zero(self):
        parser = ArgumentParser()
        group = OptimizationParams(parser)
        args = parser.parse_args(
            [
                "--density_control", "improvedgs",
                "--use_las", "0",
                "--use_rap", "0",
                "--use_gc", "0",
                "--use_absgrad", "0",
                "--use_eas", "0",
                "--use_mu", "0",
                "--gaussian_budget", "1234",
            ]
        )
        parsed = group.extract(args)
        self.assertEqual(parsed.density_control, "improvedgs")
        self.assertEqual(parsed.gaussian_budget, 1234)
        for name in (
            "use_las", "use_rap", "use_gc", "use_absgrad", "use_eas", "use_mu"
        ):
            self.assertEqual(getattr(parsed, name), 0)

    def test_pixelgs_arguments_parse_with_paper_defaults(self):
        parser = ArgumentParser()
        group = OptimizationParams(parser)
        args = parser.parse_args(
            [
                "--density_control", "pixelgs",
                "--pixelgs_depth_threshold", "0.37",
                "--densify_grad_threshold", "0.0002",
            ]
        )
        parsed = group.extract(args)
        VALIDATE_DENSITY_CONTROL(parsed)

        self.assertEqual(parsed.density_control, "pixelgs")
        self.assertAlmostEqual(parsed.pixelgs_depth_threshold, 0.37)
        self.assertAlmostEqual(parsed.densify_grad_threshold, 0.0002)
        # Improved-GS-only buffers/components must not be enabled in Pixel-GS.
        self.assertEqual(parsed.use_absgrad, 0)

    def test_pixelgs_depth_threshold_must_be_positive(self):
        parser = ArgumentParser()
        group = OptimizationParams(parser)
        for invalid in ("0", "-0.1"):
            parsed = group.extract(
                parser.parse_args(
                    [
                        "--density_control", "pixelgs",
                        "--pixelgs_depth_threshold", invalid,
                    ]
                )
            )
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "pixelgs_depth_threshold"):
                    VALIDATE_DENSITY_CONTROL(parsed)

    def test_unknown_density_control_is_rejected(self):
        parser = ArgumentParser()
        group = OptimizationParams(parser)
        parsed = group.extract(
            parser.parse_args(["--density_control", "not-a-method"])
        )
        with self.assertRaisesRegex(ValueError, "density_control"):
            VALIDATE_DENSITY_CONTROL(parsed)


if __name__ == "__main__":
    unittest.main()
