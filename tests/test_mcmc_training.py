import ast
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]


def _load_train_helpers():
    module = ast.parse((ROOT / "train.py").read_text(encoding="utf-8"))
    wanted = {
        "_validate_density_control_options",
        "_checkpoint_iteration",
        "_atomic_save_checkpoint",
        "_prune_old_checkpoints",
    }
    selected = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    namespace = {"Path": Path, "torch": torch, "os": os}
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), "train.py", "exec"),
        namespace,
    )
    return namespace


HELPERS = _load_train_helpers()


class McmcTrainingContractTests(unittest.TestCase):
    def test_mcmc_normalizes_every_improved_component_off(self):
        options = SimpleNamespace(
            density_control="mcmc",
            use_las=1,
            use_rap=1,
            use_gc=1,
            use_absgrad=1,
            use_eas=1,
            use_mu=1,
        )
        HELPERS["_validate_density_control_options"](options)
        self.assertEqual(options.density_control, "mcmc")
        for name in (
            "use_las",
            "use_rap",
            "use_gc",
            "use_absgrad",
            "use_eas",
            "use_mu",
        ):
            self.assertEqual(getattr(options, name), 0)

    def test_atomic_checkpoint_and_keep_last(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for iteration in (100, 200, 300):
                path = root / f"chkpnt{iteration}.pth"
                HELPERS["_atomic_save_checkpoint"](
                    {"iteration": iteration}, path
                )
                self.assertFalse(Path(str(path) + ".tmp").exists())

            removed = HELPERS["_prune_old_checkpoints"](root, 1)
            self.assertEqual(
                [path.name for path in removed],
                ["chkpnt100.pth", "chkpnt200.pth"],
            )
            remaining = sorted(path.name for path in root.glob("chkpnt*.pth"))
            self.assertEqual(remaining, ["chkpnt300.pth"])
            payload = torch.load(root / remaining[0], weights_only=False)
            self.assertEqual(payload["iteration"], 300)

    def test_checkpoint_filename_parser_rejects_partial_files(self):
        parse = HELPERS["_checkpoint_iteration"]
        self.assertEqual(parse("chkpnt7000.pth"), 7000)
        self.assertIsNone(parse("chkpnt7000.pth.tmp"))
        self.assertIsNone(parse("checkpoint7000.pth"))

    def test_negative_checkpoint_retention_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "non-negative"):
                HELPERS["_prune_old_checkpoints"](directory, -1)

    def test_mcmc_update_order_matches_controller_contract(self):
        source = (ROOT / "train.py").read_text(encoding="utf-8")
        start = source.index(
            "            elif mcmc_mode:", source.index("def training(")
        )
        end = source.index(
            "            else:\n                # Densification", start
        )
        section = source[start:end]
        structural = section.index("gaussians.mcmc_relocate_and_grow")
        exposure_adam = section.index("gaussians.exposure_optimizer.step")
        gaussian_adam = section.index("gaussians.optimizer.step")
        sgld = section.index("gaussians.add_mcmc_position_noise")
        self.assertLess(structural, exposure_adam)
        self.assertLess(exposure_adam, gaussian_adam)
        self.assertLess(gaussian_adam, sgld)

    def test_resume_metadata_is_written_only_after_validation(self):
        source = (ROOT / "train.py").read_text(encoding="utf-8")
        training_source = source[source.index("def training(") :]
        validate = training_source.index("validate_mcmc_resume_config(")
        write_method = training_source.index("write_mcmc_config(")
        write_legacy = training_source.index("_write_cfg_args(dataset)")
        self.assertLess(validate, write_method)
        self.assertLess(validate, write_legacy)

    def test_final_checkpoint_drops_pending_dense_gradients(self):
        source = (ROOT / "train.py").read_text(encoding="utf-8")
        self.assertIn(
            "include_gradients=(iteration < int(opt.iterations))", source
        )


if __name__ == "__main__":
    unittest.main()
