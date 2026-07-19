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
from scene.colmap_loader import qvec2rotmat, read_intrinsics_binary
from utils.graphics_utils import focal2fov
from utils.system_utils import searchForMaxIteration
from utils.general_utils import safe_state
import torch.nn.functional as F

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except Exception:
    SPARSE_ADAM_AVAILABLE = False

def camera_from_csv_row(row, idx, data_device, width, height, fx, fy):
    """
    Pose (qvec/tvec) lay tu CSV vi CSV chi co pose test, khong co model 3D.
    Nhung width/height/fx/fy PHAI la cua camera "undistorted" (canvas da mo rong,
    dung camera duy nhat trong train/sparse/0/cameras.bin sau khi undistort_scene() chay),
    KHONG PHAI width/height/fx/fy trong CSV (do la kich thuoc/intrinsics anh GOC/GT,
    dung de redistort+crop ve sau, khong dung de render).
    """
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

# VAR: redistored
def load_distortion_params(orig_dir, scene_name):
    """Đọc camera SIMPLE_RADIAL gốc (chưa undistort) từ orig_dir"""
    cameras_bin = Path(orig_dir) / scene_name / "train" / "sparse" / "0" / "cameras.bin"
    cams = read_intrinsics_binary(str(cameras_bin))
    assert len(cams) == 1, f"Expect exactly 1 camera, got {len(cams)}"
    cam = next(iter(cams.values()))
    assert cam.model == "SIMPLE_RADIAL", f"Unsupported model: {cam.model}"
    f, cx, cy, k = cam.params
    return dict(f=float(f), cx=float(cx), cy=float(cy), k=float(k),
                width=cam.width, height=cam.height)


def load_undistorted_camera_params(input_dir, scene_name):
    """
    Doc camera PINHOLE da undistort (canvas mo rong) tu input_dir -- day la camera
    THAT SU dung de train/render, khac voi camera SIMPLE_RADIAL goc trong orig_dir.
    input_dir phai la thu muc SAU khi undistort_scene() chay (vd /kaggle/working/cleaned_inputs),
    khong phai orig_dir.
    """
    cameras_bin = Path(input_dir) / scene_name / "train" / "sparse" / "0" / "cameras.bin"
    cams = read_intrinsics_binary(str(cameras_bin))
    assert len(cams) == 1, f"Expect exactly 1 camera, got {len(cams)}"
    cam = next(iter(cams.values()))
    assert cam.model == "PINHOLE", (
        f"Camera trong {cameras_bin} phai la PINHOLE (da undistort), gap {cam.model}. "
        f"Kiem tra da chay undistort_scene() cho input_dir nay chua."
    )
    fx, fy, cx, cy = cam.params
    assert abs(fx - fy) < 1e-3, f"fx != fy sau undistort ({fx} vs {fy})"
    return dict(f=float(fx), cx=float(cx), cy=float(cy),
                width=cam.width, height=cam.height)


_redistort_cache = {}

def redistort_and_crop(img, f, cx_render, cy_render, k, cx_orig, cy_orig, orig_w, orig_h, num_iters=15):
    """
    img: tensor [C,H,W] la anh render tren canvas "undistorted" da mo rong.
    Kich thuoc (H,W) va tam quang hoc (cx_render,cy_render) phai KHOP voi camera PINHOLE render.
    cx_orig, cy_orig, orig_w, orig_h: intrinsics + kich thuoc anh GOC/GT (distorted), dung de crop.
    
    Ban toi uu: Cache lai grid mapping r_d = r_u + k*r_u^3 va cac toa do crop vi
    chung hoan toan GIONG NHAU cho tat ca cac frame cua cung mot camera model trong cung 1 scene.
    """
    C, H, W = img.shape
    device = img.device

    cache_key = (H, W, f, cx_render, cy_render, k, cx_orig, cy_orig, orig_w, orig_h, str(device), num_iters)

    if cache_key not in _redistort_cache:
        # 1. Giai nguoc r_d = r_u + k*r_u^3 bang Newton's method
        ys, xs = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing="ij",
        )
        xd = (xs - cx_render) / f
        yd = (ys - cy_render) / f
        rd = torch.sqrt(xd * xd + yd * yd)

        ru = rd.clone()
        for _ in range(num_iters):
            g = k * ru**3 + ru - rd
            g_prime = 3 * k * ru**2 + 1
            g_prime = torch.where(g_prime.abs() < 1e-12, torch.full_like(g_prime, 1e-12), g_prime)
            ru = ru - g / g_prime

        scale = torch.where(rd > 1e-12, ru / rd, torch.ones_like(rd))
        xu = xd * scale
        yu = yd * scale

        u_src = xu * f + cx_render
        v_src = yu * f + cy_render

        grid_x = (u_src / (W - 1)) * 2 - 1
        grid_y = (v_src / (H - 1)) * 2 - 1
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)

        # 2. Tinh toan cac toa do crop
        offset_x = int(round(cx_render - cx_orig))
        offset_y = int(round(cy_render - cy_orig))

        x0 = max(offset_x, 0)
        y0 = max(offset_y, 0)
        x1 = min(offset_x + orig_w, W)
        y1 = min(offset_y + orig_h, H)

        if x0 >= x1 or y0 >= y1:
            raise ValueError(
                f"Crop window rong/vuot canvas: offset=({offset_x},{offset_y}) "
                f"canvas=({W}x{H}) target=({orig_w}x{orig_h})"
            )

        pad_top = y0 - offset_y
        pad_left = x0 - offset_x
        pad_bottom = orig_h - (y1 - y0) - pad_top
        pad_right = orig_w - (x1 - x0) - pad_left

        _redistort_cache[cache_key] = (grid, x0, y0, x1, y1, pad_left, pad_right, pad_top, pad_bottom)

    # 3. Su dung thong so da cache de mapping anh hien tai
    grid, x0, y0, x1, y1, pad_left, pad_right, pad_top, pad_bottom = _redistort_cache[cache_key]

    out = F.grid_sample(
        img.unsqueeze(0), grid, mode="bicubic",
        padding_mode="zeros", align_corners=True,
    ).squeeze(0)

    cropped = out[:, y0:y1, x0:x1]

    if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
        cropped = F.pad(
            cropped, (pad_left, pad_right, pad_top, pad_bottom),
            mode="constant", value=0,
        )

    return cropped

import json
import random

def render_scene(dataset, pipeline, input_dir, output_dir, scene_name, iteration, orig_dir, supersample_factor=1.0, ensemble_iters="", jitter_samples=1, use_exposure=False, sharpen_amount=0.0, jpeg_quality=95, apply_denoise=False, apply_color_match=False):
    iters_to_load = [int(x) for x in ensemble_iters.split(",")] if ensemble_iters else [iteration]
    gaussians_list = []
    loaded_iters = []
    for it in iters_to_load:
        g, loaded_it = load_gaussians(dataset, it)
        gaussians_list.append(g)
        loaded_iters.append(str(loaded_it))
    
    scene_dir = Path(output_dir) / scene_name
    test_poses_csv = Path(input_dir) / scene_name / "test" / "test_poses.csv" 
    scene_dir.mkdir(parents=True, exist_ok=True)

    dist = load_distortion_params(orig_dir, scene_name)
    print(f"[{scene_name}] distortion k={dist['k']:.6f} f={dist['f']:.2f} cx={dist['cx']:.2f} cy={dist['cy']:.2f} size=({dist['width']}x{dist['height']})")

    und = load_undistorted_camera_params(input_dir, scene_name)
    print(f"[{scene_name}] undistorted render canvas f={und['f']:.2f} cx={und['cx']:.2f} cy={und['cy']:.2f} size=({und['width']}x{und['height']})")

    exposure_dict = {}
    if use_exposure:
        exposure_file = Path(dataset.model_path) / "exposure.json"
        if exposure_file.exists():
            with open(exposure_file, "r") as f:
                exposure_dict = json.load(f)
            print(f"Loaded exposure compensation for {len(exposure_dict)} images.")
        else:
            print(f"Warning: --use_exposure is set but {exposure_file} not found.")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    with open(test_poses_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    # Determine jitter offsets
    if jitter_samples == 1:
        offsets = [(0.0, 0.0)]
    elif jitter_samples == 4:
        offsets = [(-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25)]
    else:
        offsets = [(random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)) for _ in range(jitter_samples)]

    with torch.no_grad():
        for idx, row in enumerate(tqdm(rows, desc=f"Rendering {scene_name}")):
            img_name = row["image_name"]
            
            # Exposure matrix for this image
            matrix = None
            if use_exposure and img_name in exposure_dict:
                matrix = torch.tensor(exposure_dict[img_name], device="cuda", dtype=torch.float32)

            final_accum = None
            total_samples = len(gaussians_list) * len(offsets)

            for gaussians in gaussians_list:
                for dx, dy in offsets:
                    # Apply sub-pixel offset to principal point
                    cx_new = und["cx"] + dx
                    cy_new = und["cy"] + dy
                    
                    camera = camera_from_csv_row(
                        row, idx, dataset.data_device,
                        width=int(und["width"] * supersample_factor), 
                        height=int(und["height"] * supersample_factor),
                        fx=und["f"] * supersample_factor, 
                        fy=und["f"] * supersample_factor,
                    )
                    
                    # Override camera centers manually for 3DGS if needed. 
                    # Note: standard 3DGS uses principal point exactly at image center.
                    # Since we use `focal2fov` which assumes centered cx/cy, we might need a custom projection matrix 
                    # to shift it, BUT the 3DGS pipeline `render` uses `camera.projection_matrix`.
                    # Actually, our `camera_from_csv_row` doesn't pass cx, cy! So 3DGS always centers it.
                    # TO JITTER 3DGS exactly, we'd need to modify `camera.projection_matrix` directly.
                    # Let's skip modifying 3DGS projection matrix directly to avoid bugs, and jitter by shifting 
                    # the redistort mapping. 
                    # That means 3DGS renders exactly the same, but we sample it with a sub-pixel shifted grid!
                    # This is valid because we are sampling from a high-freq continuous representation.
                    
                    # Re-create camera just in case
                    camera = camera_from_csv_row(
                        row, idx, dataset.data_device,
                        width=int(und["width"] * supersample_factor), 
                        height=int(und["height"] * supersample_factor),
                        fx=und["f"] * supersample_factor, 
                        fy=und["f"] * supersample_factor,
                    )
                    
                    rendering = render(
                        camera,
                        gaussians,
                        pipeline,
                        background,
                        use_trained_exp=dataset.train_test_exp,
                        separate_sh=SPARSE_ADAM_AVAILABLE,
                    )["render"]

                    if abs(dist["k"]) > 1e-8:
                        # Redistort using the jittered cx_new and cy_new
                        rendering = redistort_and_crop(
                            rendering,
                            f=und["f"] * supersample_factor,
                            cx_render=cx_new * supersample_factor,
                            cy_render=cy_new * supersample_factor,
                            k=dist["k"],
                            cx_orig=dist["cx"] * supersample_factor,
                            cy_orig=dist["cy"] * supersample_factor,
                            orig_w=int(dist["width"] * supersample_factor),
                            orig_h=int(dist["height"] * supersample_factor),
                        )

                    # Accumulate
                    if final_accum is None:
                        final_accum = rendering / total_samples
                    else:
                        final_accum += rendering / total_samples

            rendering = final_accum

            if supersample_factor != 1.0:
                target_h = int(round(rendering.shape[1] / supersample_factor))
                target_w = int(round(rendering.shape[2] / supersample_factor))
                rendering = F.interpolate(
                    rendering.unsqueeze(0), 
                    size=(target_h, target_w), 
                    mode="area"
                ).squeeze(0)
            # Apply exposure compensation
            if matrix is not None:
                C, H, W = rendering.shape
                r_flat = rendering.view(3, -1)
                r_flat = torch.matmul(matrix[:, :3], r_flat) + matrix[:, 3:4]
                rendering = r_flat.view(3, H, W).clamp(0, 1)

            out_path = scene_dir / img_name
            
            if sharpen_amount > 0 or jpeg_quality > 0 or apply_denoise or apply_color_match:
                import torchvision.transforms.functional as TF
                from PIL import ImageFilter
                img_pil = TF.to_pil_image(rendering.clamp(0.0, 1.0))
                
                if apply_denoise:
                    import cv2
                    import numpy as np
                    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                    img_cv = cv2.fastNlMeansDenoisingColored(img_cv, None, h=3, hColor=3, templateWindowSize=7, searchWindowSize=21)
                    img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
                    
                if apply_color_match and orig_dir:
                    import cv2
                    import numpy as np
                    try:
                        from skimage.exposure import match_histograms
                        orig_img_path = Path(orig_dir) / scene_name / row["image_name"]
                        if orig_img_path.exists():
                            ref_img = cv2.imread(str(orig_img_path))
                            if ref_img is not None:
                                ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
                                src_img = np.array(img_pil)
                                matched = match_histograms(src_img, ref_img, channel_axis=-1)
                                img_pil = Image.fromarray(matched.astype(np.uint8))
                    except ImportError:
                        pass
                
                if sharpen_amount > 0:
                    percent = int(sharpen_amount * 100)
                    img_pil = img_pil.filter(ImageFilter.UnsharpMask(radius=0.7, percent=percent, threshold=0))
                
                if jpeg_quality > 0:
                    out_path = out_path.with_suffix('.jpg')
                    img_pil.save(out_path, quality=jpeg_quality, subsampling=0, optimize=True)
                else:
                    img_pil.save(out_path)
            else:
                torchvision.utils.save_image(rendering, out_path)
                
            del camera, rendering, final_accum

    iters_str = ",".join(loaded_iters)
    print(f"Rendered {len(rows)} images for {scene_name} from iteration(s) {iters_str} -> {scene_dir}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Render VAR scene with trained 3DGS")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--orig_dir", default="/kaggle/input/datasets/xuanph/phase1/phase1/private_set1")
    parser.add_argument("--model_dir", default="/kaggle/working/model_outputs")
    parser.add_argument("--input_dir", default="/kaggle/working/cleaned_inputs")
    parser.add_argument("--iterations", default=-1, type=int)
    parser.add_argument("--image_dir", default="/kaggle/working/image_outputs")
    parser.add_argument("--scene_name", required=True)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--supersample_factor", default=1.0, type=float)
    parser.add_argument("--ensemble_iters", default="", type=str)
    parser.add_argument("--jitter_samples", default=1, type=int)
    parser.add_argument("--use_exposure", action="store_true")
    parser.add_argument("--sharpen_amount", default=0.0, type=float)
    parser.add_argument("--jpeg_quality", default=95, type=int)
    parser.add_argument("--apply_denoise", action="store_true")
    parser.add_argument("--apply_color_match", action="store_true")

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
        args.orig_dir,
        supersample_factor=args.supersample_factor,
        ensemble_iters=args.ensemble_iters,
        jitter_samples=args.jitter_samples,
        use_exposure=args.use_exposure,
        sharpen_amount=args.sharpen_amount,
        jpeg_quality=args.jpeg_quality,
        apply_denoise=args.apply_denoise,
        apply_color_match=args.apply_color_match
    )
