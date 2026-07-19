from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RASTERIZER = ROOT / "submodules" / "diff-gaussian-rasterization"
PATCH = ROOT / "patches" / "improved-gs-rasterizer.patch"


class McmcRasterizerPatchTests(unittest.TestCase):
    def test_build_system_compiles_relocation_kernel(self):
        setup_source = (RASTERIZER / "setup.py").read_text(encoding="utf-8")
        setup_tree = ast.parse(setup_source)
        setup_strings = {
            node.value for node in ast.walk(setup_tree) if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
        }
        self.assertIn("cuda_rasterizer/utils.cu", setup_strings)

        cmake = (RASTERIZER / "CMakeLists.txt").read_text(encoding="utf-8")
        self.assertIn("cuda_rasterizer/utils.cu", cmake)
        self.assertIn("cuda_rasterizer/utils.h", cmake)

    def test_extension_and_python_api_export_compute_relocation(self):
        extension = (RASTERIZER / "ext.cpp").read_text(encoding="utf-8")
        header = (RASTERIZER / "rasterize_points.h").read_text(encoding="utf-8")
        source = (RASTERIZER / "rasterize_points.cu").read_text(encoding="utf-8")
        python_api = (
            RASTERIZER / "diff_gaussian_rasterization" / "__init__.py"
        ).read_text(encoding="utf-8")

        self.assertIn('m.def("compute_relocation", &ComputeRelocationCUDA)', extension)
        self.assertIn("ComputeRelocationCUDA", header)
        self.assertIn("UTILS::ComputeRelocation", source)
        self.assertIn("def compute_relocation(", python_api)
        self.assertIn("N.to(dtype=torch.int32).clamp(min=1, max=n_max - 1)", python_api)

    def test_kernel_contains_paper_opacity_and_scale_updates(self):
        kernel = (RASTERIZER / "cuda_rasterizer" / "utils.cu").read_text(
            encoding="utf-8"
        )
        self.assertIn("1.0f - powf(1.0f - old_opacity", kernel)
        self.assertIn("old_opacity / denominator", kernel)
        self.assertIn("binoms[(i - 1) * n_max + k]", kernel)
        self.assertIn("multiplicity = multiplicity < 1 ? 1 : multiplicity", kernel)
        self.assertIn("multiplicity > n_max - 1 ? n_max - 1", kernel)

    def test_cumulative_parent_patch_tracks_relocation_sources(self):
        patch = PATCH.read_text(encoding="utf-8")
        for path in (
            "cuda_rasterizer/utils.cu",
            "cuda_rasterizer/utils.h",
            "diff_gaussian_rasterization/__init__.py",
            "rasterize_points.cu",
            "rasterize_points.h",
            "setup.py",
            "CMakeLists.txt",
            "ext.cpp",
        ):
            with self.subTest(path=path):
                self.assertIn("diff --git a/{} b/{}".format(path, path), patch)
        self.assertIn('m.def("compute_relocation", &ComputeRelocationCUDA)', patch)


if __name__ == "__main__":
    unittest.main()
