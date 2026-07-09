import os
import sys
import subprocess
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser(description="Train all scenes")
    parser.add_argument("--input_dir", default="/kaggle/working/cleaned_inputs")
    parser.add_argument("--model_dir", default="/kaggle/working/model_outputs")
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30000])
    parser.add_argument("--extra_args", nargs="*", default=[])
    parser.add_argument("--subset", nargs="+", default=[])
    parser.add_argument("--cap_max", type=int, default=-1)
    args = parser.parse_args()

    # sort 
    scenes = sorted([
        d for d in os.listdir(args.input_dir)
        if os.path.isdir(os.path.join(args.input_dir, d))
    ])
    print(f"Found {len(scenes)} scenes: {scenes}")

    # using subset scene if exsist
    if args.subset:
        missing = [s for s in args.subset if s not in scenes]
        if missing:
            print(f"!!! Warning: subset scenes not found in input_dir: {missing}")
        scenes = [s for s in scenes if s in args.subset]
        print(f"Filtered to subset ({len(scenes)}): {scenes}")

    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    failed_scenes = []

    for i, scene in enumerate(scenes):
        model_path = os.path.join(args.model_dir, scene)

        print(f"\n=== [{i+1}/{len(scenes)}] Training scene: {scene} ===")
        cmd = [
            sys.executable, "train_scene.py",
            "--input_dir", args.input_dir,
            "--model_dir", args.model_dir,
            "--scene_name", scene,
            "--iterations", str(args.iterations),
            "--test_iterations", *map(str, args.test_iterations),
            "--save_iterations", *map(str, args.save_iterations),
            "--cap_max", str(args.cap_max)
        ] + args.extra_args

        ret = subprocess.run(cmd, env=env)
        if ret.returncode != 0:
            print(f"!!! Scene {scene} failed (code {ret.returncode}), continuing...")
            failed_scenes.append(scene)

    print(f"\nDone. {len(scenes) - len(failed_scenes)}/{len(scenes)} scenes succeeded.")
    if failed_scenes:
        print(f"Failed scenes: {failed_scenes}")