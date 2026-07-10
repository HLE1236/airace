import csv
import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from scene.cameras import Camera
from scene.colmap_loader import qvec2rotmat
from utils.graphics_utils import focal2fov
from utils.system_utils import searchForMaxIteration
from utils.general_utils import safe_state

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except Exception:
    SPARSE_ADAM_AVAILABLE = False

def camera_from_csv_row(row, idx, data_device):
    width = int(float(row["width"]))
    height = int(float(row["height"]))
    fx = float(row["fx"])
    fy = float(row["fy"])

    qvec = np.array(
        [float(row["qw"]), float(row["qx"]), float(row["qy"]), float(row["qz"])],
        dtype=np.float64,
    )

    # The competition README calls tx/ty/tz "camera position", but the released
    # public poses match COLMAP tvec distribution. 3DGS expects COLMAP-style
    # world-to-camera translation here.
    tvec = np.array([float(row["tx"]), float(row["ty"]), float(row["tz"])], dtype=np.float64)
    rotation_world_to_camera = qvec2rotmat(qvec)

    dummy = Image.new("RGB", (width, height), (0, 0, 0))
    return Camera(
        resolution=(width, height),
        colmap_id=idx,
        R=rotation_world_to_camera.T,
        T=tvec,
        FoVx=focal2fov(fx, width),
        FoVy=focal2fov(fy, height),
        depth_params=None,
        image=dummy,
        invdepthmap=None,
        image_name=Path(row["image_name"]).name,
        uid=idx,
        data_device=data_device,
    )

def load_gaussians(dataset, iteration):
    gaussians = GaussianModel(dataset.sh_degree)
    loaded_iter = searchForMaxIteration(os.path.join(dataset.model_path, "point_cloud")) if iteration == -1 else iteration
    ply_path = os.path.join(
        dataset.model_path,
        "point_cloud",
        f"iteration_{loaded_iter}",
        "point_cloud.ply",
    )
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"Cannot find trained point cloud: {ply_path}")
    print(f"Loading trained model at iteration {loaded_iter}")
    gaussians.load_ply(ply_path, dataset.train_test_exp)
    return gaussians, loaded_iter

def render_scene(dataset, pipeline, input_dir ,output_dir, scene_name, iteration):
    gaussians, loaded_iter = load_gaussians(dataset, iteration)
    scene_dir = Path(output_dir) / scene_name
    test_poses_csv = Path(input_dir) / scene_name / "test" / "test_poses.csv" 
    scene_dir.mkdir(parents=True, exist_ok=True)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    with open(test_poses_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    with torch.no_grad():
        for idx, row in enumerate(tqdm(rows, desc=f"Rendering {scene_name}")):
            camera = camera_from_csv_row(row, idx, dataset.data_device)
            rendering = render(
                camera,
                gaussians,
                pipeline,
                background,
                use_trained_exp=dataset.train_test_exp,
                separate_sh=SPARSE_ADAM_AVAILABLE,
            )["render"]
            out_path = scene_dir / row["image_name"]
            torchvision.utils.save_image(rendering, out_path)
            del camera, rendering

    print(f"Rendered {len(rows)} images for {scene_name} from iteration {loaded_iter} -> {scene_dir}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Render VAR scene with trained 3DGS")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--model_dir", default="/kaggle/working/model_outputs")
    parser.add_argument("--input_dir", default="/kaggle/working/cleaned_inputs")
    parser.add_argument("--iterations", default=-1, type=int)
    parser.add_argument("--image_dir", default="/kaggle/working/image_outputs")
    parser.add_argument("--scene_name", required=True)
    parser.add_argument("--quiet", action="store_true")

    args = get_combined_args(parser)

    safe_state(args.quiet)

    os.makedirs(args.image_dir, exist_ok=True)

    render_scene(
        model.extract(args),
        pipeline.extract(args),
        args.input_dir,
        args.image_dir,
        args.scene_name,
        args.iterations,
    )