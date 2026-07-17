import os
import sys
import subprocess
from argparse import ArgumentParser
from pathlib import Path

if __name__ == "__main__":
    parser = ArgumentParser(description="Render all scenes with trained 3DGS")
    parser.add_argument("--orig_dir", default="/kaggle/input/datasets/xuanph/phase1/phase1/private_set1")
    parser.add_argument("--model_dir", default="/kaggle/working/model_outputs")
    parser.add_argument("--input_dir", default="/kaggle/working/cleaned_inputs")
    parser.add_argument("--image_dir", default="/kaggle/working/image_outputs")
    parser.add_argument("--iterations", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--subset", nargs="+", default=[])
    parser.add_argument("--extra_args", nargs="*", default=[])
    
    # Advanced rendering options
    parser.add_argument("--supersample_factor", default=1.0, type=float, help="Scale factor for supersampling")
    parser.add_argument("--ensemble_iters", default="", type=str, help="Comma-separated iterations to average (e.g. 29000,30000)")
    parser.add_argument("--jitter_samples", default=1, type=int, help="Number of sub-pixel jitter samples for SSAA")
    parser.add_argument("--use_exposure", action="store_true", help="Apply exposure compensation from exposure.json")
    parser.add_argument("--sharpen_amount", default=0.0, type=float, help="UnsharpMask percent (e.g. 0.3 for 30%)")
    parser.add_argument("--jpeg_quality", default=0, type=int, help="Save as JPEG with this quality and 4:4:4. If 0, saves as PNG/default.")
    
    args = parser.parse_args()

    scenes = sorted([
        d for d in os.listdir(args.input_dir)
        if os.path.isdir(os.path.join(args.input_dir, d))
    ])
    print(f"Found {len(scenes)} scenes: {scenes}")

    if args.subset:
        missing = [s for s in args.subset if s not in scenes]
        if missing:
            print(f"!!! Warning: subset scenes not found in input_dir: {missing}")
        scenes = [s for s in scenes if s in args.subset]
        print(f"Filtered to subset ({len(scenes)}): {scenes}")

    failed_scenes = []

    for i, scene in enumerate(scenes):
        scene_out = Path(args.image_dir) / scene
        if args.skip_existing and scene_out.exists() and any(scene_out.iterdir()):
            print(f"[{i+1}/{len(scenes)}] Skip {scene} (already rendered)")
            continue

        model_path = os.path.join(args.model_dir, scene)
        cfg_path = os.path.join(model_path, "cfg_args")
        if not os.path.exists(cfg_path):
            print(f"!!! Skip {scene}: cfg_args not found ({cfg_path}), scene chưa train xong")
            failed_scenes.append(scene)
            continue

        print(f"\n=== [{i+1}/{len(scenes)}] Rendering scene: {scene} ===")
        cmd = [
            sys.executable, "render_scene.py",
            "--model_dir", args.model_dir,
            "--input_dir", args.input_dir,
            "--image_dir", args.image_dir,
            "--scene_name", scene,
            "--iterations", str(args.iterations),
            "--orig_dir", args.orig_dir,
        ]
        if args.quiet:
            cmd.append("--quiet")
        if args.supersample_factor != 1.0:
            cmd.extend(["--supersample_factor", str(args.supersample_factor)])
        if args.ensemble_iters:
            cmd.extend(["--ensemble_iters", args.ensemble_iters])
        if args.jitter_samples > 1:
            cmd.extend(["--jitter_samples", str(args.jitter_samples)])
        if args.use_exposure:
            cmd.append("--use_exposure")
        if args.sharpen_amount > 0:
            cmd.extend(["--sharpen_amount", str(args.sharpen_amount)])
        if args.jpeg_quality > 0:
            cmd.extend(["--jpeg_quality", str(args.jpeg_quality)])
            
        cmd += args.extra_args

        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            print(f"!!! Scene {scene} failed (code {ret.returncode}), continuing...")
            failed_scenes.append(scene)

    print(f"\nDone. {len(scenes) - len(failed_scenes)}/{len(scenes)} scenes succeeded.")
    if failed_scenes:
        print(f"Failed scenes: {failed_scenes}")