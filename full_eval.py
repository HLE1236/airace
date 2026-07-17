#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import subprocess
from argparse import ArgumentParser
import time
from pathlib import Path

mipnerf360_outdoor_scenes = ["bicycle", "flowers", "garden", "stump", "treehill"]
mipnerf360_indoor_scenes = ["room", "counter", "kitchen", "bonsai"]
tanks_and_temples_scenes = ["truck", "train"]
deep_blending_scenes = ["drjohnson", "playroom"]

def train_scenes(args, common_args):
    timing = {}
    
    start_time = time.time()
    for scene in mipnerf360_outdoor_scenes:
        source = Path(args.mipnerf360) / scene
        cmd = f"python train.py -s {source} -i images_4 -m {Path(args.output_path) / scene} {common_args}"
        subprocess.run(cmd, shell=True, check=True)
    for scene in mipnerf360_indoor_scenes:
        source = Path(args.mipnerf360) / scene
        cmd = f"python train.py -s {source} -i images_2 -m {Path(args.output_path) / scene} {common_args}"
        subprocess.run(cmd, shell=True, check=True)
    timing["m360"] = (time.time() - start_time) / 60.0

    start_time = time.time()
    for scene in tanks_and_temples_scenes:
        source = Path(args.tanksandtemples) / scene
        cmd = f"python train.py -s {source} -m {Path(args.output_path) / scene} {common_args}"
        subprocess.run(cmd, shell=True, check=True)
    timing["tandt"] = (time.time() - start_time) / 60.0

    start_time = time.time()
    for scene in deep_blending_scenes:
        source = Path(args.deepblending) / scene
        cmd = f"python train.py -s {source} -m {Path(args.output_path) / scene} {common_args}"
        subprocess.run(cmd, shell=True, check=True)
    timing["db"] = (time.time() - start_time) / 60.0
    
    return timing

def render_scenes(args, common_args):
    all_scenes = mipnerf360_outdoor_scenes + mipnerf360_indoor_scenes + tanks_and_temples_scenes + deep_blending_scenes
    all_sources = []
    all_sources.extend([Path(args.mipnerf360) / s for s in mipnerf360_outdoor_scenes + mipnerf360_indoor_scenes])
    all_sources.extend([Path(args.tanksandtemples) / s for s in tanks_and_temples_scenes])
    all_sources.extend([Path(args.deepblending) / s for s in deep_blending_scenes])
    
    for scene, source in zip(all_scenes, all_sources):
        cmd7k = f"python render.py --iteration 7000 -s {source} -m {Path(args.output_path) / scene} {common_args}"
        cmd30k = f"python render.py --iteration 30000 -s {source} -m {Path(args.output_path) / scene} {common_args}"
        subprocess.run(cmd7k, shell=True, check=True)
        subprocess.run(cmd30k, shell=True, check=True)

def run_metrics(args, all_scenes):
    scenes_string = " ".join([f'"{Path(args.output_path) / scene}"' for scene in all_scenes])
    cmd = f"python metrics.py -m {scenes_string}"
    subprocess.run(cmd, shell=True, check=True)

def main():
    parser = ArgumentParser(description="Full evaluation script parameters")
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--skip_rendering", action="store_true")
    parser.add_argument("--skip_metrics", action="store_true")
    parser.add_argument("--output_path", default="./eval")
    parser.add_argument("--use_depth", action="store_true")
    parser.add_argument("--use_expcomp", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--aa", action="store_true")

    args, _ = parser.parse_known_args()

    all_scenes = mipnerf360_outdoor_scenes + mipnerf360_indoor_scenes + tanks_and_temples_scenes + deep_blending_scenes

    if not args.skip_training or not args.skip_rendering:
        parser.add_argument('--mipnerf360', "-m360", required=True, type=str)
        parser.add_argument("--tanksandtemples", "-tat", required=True, type=str)
        parser.add_argument("--deepblending", "-db", required=True, type=str)
        args = parser.parse_args()

    Path(args.output_path).mkdir(parents=True, exist_ok=True)

    if not args.skip_training:
        common_args = " --disable_viewer --quiet --eval --test_iterations -1 "
        if args.aa:
            common_args += " --antialiasing "
        if args.use_depth:
            common_args += " -d depths2/ "
        if args.use_expcomp:
            common_args += " --exposure_lr_init 0.001 --exposure_lr_final 0.0001 --exposure_lr_delay_steps 5000 --exposure_lr_delay_mult 0.001 --train_test_exp "
        if args.fast:
            common_args += " --optimizer_type sparse_adam "

        timing = train_scenes(args, common_args)
        
        with open(Path(args.output_path) / "timing.txt", 'w') as file:
            file.write(f"m360: {timing['m360']:.2f} minutes \n tandt: {timing['tandt']:.2f} minutes \n db: {timing['db']:.2f} minutes\n")

    if not args.skip_rendering:
        common_args = " --quiet --eval --skip_train"
        if args.aa:
            common_args += " --antialiasing "
        if args.use_expcomp:
            common_args += " --train_test_exp "
        
        render_scenes(args, common_args)

    if not args.skip_metrics:
        run_metrics(args, all_scenes)

if __name__ == "__main__":
    main()
