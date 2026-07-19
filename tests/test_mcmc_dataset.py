import ast
import unittest
from collections import namedtuple
from pathlib import Path

import numpy as np

from utils.sh_utils import SH2RGB


ROOT = Path(__file__).resolve().parents[1]
BasicPointCloud = namedtuple("BasicPointCloud", ["points", "colors", "normals"])


def _load_random_point_cloud_helper():
    module = ast.parse(
        (ROOT / "scene" / "dataset_readers.py").read_text(encoding="utf-8")
    )
    selected = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_mcmc_random_point_cloud"
    ]
    namespace = {
        "np": np,
        "SH2RGB": SH2RGB,
        "BasicPointCloud": BasicPointCloud,
    }
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), "dataset_readers.py", "exec"),
        namespace,
    )
    return namespace["_mcmc_random_point_cloud"]


RANDOM_POINT_CLOUD = _load_random_point_cloud_helper()


class McmcDatasetInitializationTests(unittest.TestCase):
    def test_random_cloud_is_deterministic_and_uses_official_bounds(self):
        first = RANDOM_POINT_CLOUD(radius=2.0, num_points=100, seed=7)
        second = RANDOM_POINT_CLOUD(radius=2.0, num_points=100, seed=7)
        different = RANDOM_POINT_CLOUD(radius=2.0, num_points=100, seed=8)
        self.assertTrue(np.array_equal(first.points, second.points))
        self.assertTrue(np.array_equal(first.colors, second.colors))
        self.assertFalse(np.array_equal(first.points, different.points))
        self.assertEqual(first.points.shape, (100, 3))
        self.assertTrue(np.all(first.points >= -6.0))
        self.assertTrue(np.all(first.points <= 6.0))
        self.assertTrue(np.all(first.normals == 0.0))

    def test_random_cloud_rejects_invalid_inputs(self):
        with self.assertRaisesRegex(ValueError, "positive radius"):
            RANDOM_POINT_CLOUD(radius=0.0)
        with self.assertRaisesRegex(ValueError, "positive"):
            RANDOM_POINT_CLOUD(radius=1.0, num_points=0)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            RANDOM_POINT_CLOUD(radius=1.0, seed=-1)

    def test_scene_keeps_random_init_opt_in_to_mcmc(self):
        source = (ROOT / "scene" / "__init__.py").read_text(encoding="utf-8")
        self.assertIn('getattr(args, "mcmc_enabled", False)', source)
        self.assertIn('else "sfm"', source)


if __name__ == "__main__":
    unittest.main()
