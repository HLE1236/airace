import ast
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]


def _load_pixelgs_config_helpers():
    """Load the pure config helpers without importing train.py's CUDA stack."""
    module = ast.parse((ROOT / "train.py").read_text(encoding="utf-8"))
    selected = []
    for node in module.body:
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "_PIXELGS_CONFIG_KEYS"
                for target in node.targets
            )
        ):
            selected.append(node)
        elif (
            isinstance(node, ast.FunctionDef)
            and node.name
            in {"_build_pixelgs_config", "_validate_pixelgs_resume_config"}
        ):
            selected.append(node)

    namespace = {}
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), "train.py", "exec"),
        namespace,
    )
    return namespace["_build_pixelgs_config"], namespace[
        "_validate_pixelgs_resume_config"
    ]


BUILD_CONFIG, VALIDATE_RESUME = _load_pixelgs_config_helpers()


def _optimization_args():
    return SimpleNamespace(
        density_control="pixelgs",
        pixelgs_depth_threshold=0.37,
        densify_grad_threshold=0.0002,
        percent_dense=0.01,
        densify_from_iter=500,
        densify_until_iter=15_000,
        densification_interval=100,
        opacity_reset_interval=3_000,
    )


class PixelGsConfigTests(unittest.TestCase):
    def test_config_records_cap_and_seed(self):
        config = BUILD_CONFIG(_optimization_args(), seed=7, cap_max=6_000_000)
        self.assertEqual(config["density_control"], "pixelgs")
        self.assertEqual(config["cap_max"], 6_000_000)
        self.assertEqual(config["seed"], 7)
        self.assertAlmostEqual(config["pixelgs_depth_threshold"], 0.37)

    def test_matching_resume_config_is_accepted(self):
        config = BUILD_CONFIG(_optimization_args(), seed=0, cap_max=6_000_000)
        VALIDATE_RESUME(
            {"density_control": "pixelgs", "resume_config": dict(config)},
            config,
        )

    def test_resume_mismatch_is_rejected_with_differing_key(self):
        current = BUILD_CONFIG(_optimization_args(), seed=0, cap_max=6_000_000)
        saved = dict(current)
        saved["pixelgs_depth_threshold"] = 0.5

        with self.assertRaisesRegex(ValueError, "pixelgs_depth_threshold"):
            VALIDATE_RESUME(
                {"density_control": "pixelgs", "resume_config": saved},
                current,
            )

    def test_non_pixelgs_checkpoint_is_rejected(self):
        current = BUILD_CONFIG(_optimization_args(), seed=0, cap_max=6_000_000)
        with self.assertRaisesRegex(ValueError, "density_control pixelgs"):
            VALIDATE_RESUME(
                {"density_control": "improvedgs", "resume_config": current},
                current,
            )


if __name__ == "__main__":
    unittest.main()
