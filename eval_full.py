import os
import sys
import json
import subprocess
from argparse import ArgumentParser
from pathlib import Path

if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluate all scenes")
    parser.add_argument("--input_dir", default="/kaggle/working/cleaned_inputs")
    parser.add_argument("--image_dir", default="/kaggle/working/image_outputs")
    parser.add_argument("--eval_dir", default="/kaggle/working/eval_outputs")
    parser.add_argument("--psnr_max", type=float, default=40.0)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--subset", nargs="+", default=[])
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
    results = []

    for i, scene in enumerate(scenes):
        out_json = Path(args.eval_dir) / f"{scene}.json"
        if args.skip_existing and out_json.exists():
            print(f"[{i+1}/{len(scenes)}] Skip {scene} (already evaluated)")
            with open(out_json) as f:
                results.append(json.load(f))
            continue

        sample_path = os.path.join(args.image_dir, scene)
        if not os.path.exists(sample_path):
            print(f"!!! Skip {scene}: sample_path not found ({sample_path}), chưa render")
            failed_scenes.append(scene)
            continue

        print(f"\n=== [{i+1}/{len(scenes)}] Evaluating scene: {scene} ===")
        cmd = [
            sys.executable, "eval_scene.py",
            "--input_dir", args.input_dir,
            "--image_dir", args.image_dir,
            "--eval_dir", args.eval_dir,
            "--scene_name", scene,
            "--psnr_max", str(args.psnr_max),
        ]

        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            print(f"!!! Scene {scene} failed (code {ret.returncode}), continuing...")
            failed_scenes.append(scene)
            continue

        with open(out_json) as f:
            results.append(json.load(f))

    if results:
        avg = {
            "num_scenes": len(results),
            "SSIM": sum(r["SSIM"] for r in results) / len(results),
            "PSNR": sum(r["PSNR"] for r in results) / len(results),
            "LPIPS": sum(r["LPIPS"] for r in results) / len(results),
            "weighted_score": sum(r["weighted_score"] for r in results) / len(results),
        }
        print("\n=== Summary ===")
        print(json.dumps(avg, indent=2))

        os.makedirs(args.eval_dir, exist_ok=True)
        with open(os.path.join(args.eval_dir, "summary.json"), "w") as f:
            json.dump({"per_scene": results, "average": avg}, f, indent=2)

    print(f"\nDone. {len(scenes) - len(failed_scenes)}/{len(scenes)} scenes succeeded.")
    if failed_scenes:
        print(f"Failed scenes: {failed_scenes}")