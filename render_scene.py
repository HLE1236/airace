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

def render_scene(dataset, pipeline, input_dir ,output_dir, scene_name, iteration, orig_dir, supersample_factor=1.0):
    gaussians, loaded_iter = load_gaussians(dataset, iteration)
    scene_dir = Path(output_dir) / scene_name
    test_poses_csv = Path(input_dir) / scene_name / "test" / "test_poses.csv" 
    scene_dir.mkdir(parents=True, exist_ok=True)

    #VAR: load cameras intrinsic for distortion factor (anh GOC, truoc undistort)
    dist = load_distortion_params(orig_dir, scene_name)
    print(f"[{scene_name}] distortion k={dist['k']:.6f} f={dist['f']:.2f} "
          f"cx={dist['cx']:.2f} cy={dist['cy']:.2f} size=({dist['width']}x{dist['height']})")

    # VAR: camera PINHOLE da undistort (canvas mo rong) -- dung camera nay de RENDER,
    # khong dung width/height/fx/fy trong CSV (do la kich thuoc GT goc).
    und = load_undistorted_camera_params(input_dir, scene_name)
    print(f"[{scene_name}] undistorted render canvas f={und['f']:.2f} "
          f"cx={und['cx']:.2f} cy={und['cy']:.2f} size=({und['width']}x{und['height']})")
    

    # #====================================================================================================================
    # print(f"[{scene_name}] dist: k={dist['k']:.8f} f={dist['f']:.3f} "
    #     f"size=({dist['width']}x{dist['height']})")
    # print(f"[{scene_name}] und : f={und['f']:.3f} "
    #     f"size=({und['width']}x{und['height']})")

    # # So sanh f_undist vs f_orig -- thuat toan redistort_and_crop dang GIA DINH 2 gia tri
    # # nay xap xi bang nhau (chi lech cx,cy do canvas mo rong). Neu lech nhieu, do cong
    # # tinh ra se bi sai ty le.
    # f_diff_pct = abs(und['f'] - dist['f']) / dist['f'] * 100
    # print(f"f_undist vs f_orig: diff={und['f']-dist['f']:+.3f}  ({f_diff_pct:.2f}%)")

    # # Ban kinh chuan hoa tai GOC anh (worst case that su, khong phai mep giua canh)
    # # vi diem xa tam quang hoc nhat luon nam o 4 goc, khong phai mep ngang/doc.
    # rd_edge_mid = (dist['width'] / 2) / dist['f']          # mep giua canh ngang (cu, thieu)
    # rd_corner = np.sqrt((dist['width'] / 2) ** 2 + (dist['height'] / 2) ** 2) / dist['f']  # goc anh

    # for label, rd in [("mep giua canh", rd_edge_mid), ("goc anh (worst case)", rd_corner)]:
    #     delta = dist['k'] * rd ** 3
    #     pct = abs(delta) / rd * 100
    #     print(f"  rd={rd:.4f} tai [{label}]: k*rd^3={delta:.6f}  (lech {pct:.2f}% so voi rd)")

    # # Kiem tra vung invalid (chi xay ra khi k < 0): neu rd_corner > rd_max, nghia la
    # # 4 goc anh GOC nam trong vung KHONG CO NGHIEM THAT khi redistort -- day la vung
    # # se bi mat/den neu code xu ly dung (giong het ảnh 2 cua experiment_distort.py).
    # if dist['k'] < 0:
    #     ru_max = np.sqrt(-1.0 / (3 * dist['k']))
    #     rd_max = ru_max + dist['k'] * ru_max ** 3
    #     print(f"k<0 -> rd_max (nguong co nghiem)={rd_max:.4f}")
    #     print(f"  rd_corner ({rd_corner:.4f}) {'VUOT NGUONG -> co vung invalid o goc anh' if rd_corner > rd_max else 'trong nguong, khong co vung invalid'}")
    # else:
    #     print("k >= 0 -> khong co vung invalid (chi xay ra khi k<0)")

    # #=============================================================================================================

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    with open(test_poses_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    with torch.no_grad():
        for idx, row in enumerate(tqdm(rows, desc=f"Rendering {scene_name}")):
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

            # if idx == 0:
            #     # VAR-DEBUG: ve luoi len anh render TRUOC khi redistort, de kiem tra
            #     # warp phi tuyen co thuc su xay ra hay chi la tinh tien (crop offset).
            #     debug_grid = rendering.clone()
            #     C, Hc, Wc = debug_grid.shape
            #     step = 50
            #     for gx in range(0, Wc, step):
            #         debug_grid[:, :, gx] = torch.tensor([1.0, 0.0, 0.0], device=debug_grid.device).view(3, 1)
            #     for gy in range(0, Hc, step):
            #         debug_grid[:, gy, :] = torch.tensor([1.0, 0.0, 0.0], device=debug_grid.device).view(3, 1)

            #     debug_grid_after = redistort_and_crop(
            #         debug_grid,
            #         f=und["f"], cx_render=und["cx"], cy_render=und["cy"],
            #         k=dist["k"],
            #         cx_orig=dist["cx"], cy_orig=dist["cy"],
            #         orig_w=dist["width"], orig_h=dist["height"],
            #     )
            #     torchvision.utils.save_image(debug_grid, "/kaggle/working/debug_grid_before.png")
            #     torchvision.utils.save_image(debug_grid_after, "/kaggle/working/debug_grid_after.png")

            # VAR: redistort tren canvas mo rong (dung intrinsics cua chinh canvas do:
            # und["f"], und["cx"], und["cy"]) roi crop ve dung kich thuoc GT goc
            # (dist["width"], dist["height"]) bang offset giua 2 tam quang hoc --
            # giong het pattern trong experiment_distort.py.
            if abs(dist["k"]) > 1e-8:
                # rendering_before = rendering.clone()
                rendering = redistort_and_crop(
                    rendering,
                    f=und["f"] * supersample_factor,
                    cx_render=und["cx"] * supersample_factor,
                    cy_render=und["cy"] * supersample_factor,
                    k=dist["k"],
                    cx_orig=dist["cx"] * supersample_factor,
                    cy_orig=dist["cy"] * supersample_factor,
                    orig_w=int(dist["width"] * supersample_factor),
                    orig_h=int(dist["height"] * supersample_factor),
                )

            if supersample_factor != 1.0:
                target_h = int(round(rendering.shape[1] / supersample_factor))
                target_w = int(round(rendering.shape[2] / supersample_factor))
                rendering = F.interpolate(
                    rendering.unsqueeze(0), 
                    size=(target_h, target_w), 
                    mode="area"
                ).squeeze(0)
                # # THEMMM
                # if idx == 0:
                #     # crop rendering_before ve cung kich thuoc de so sanh cho cong bang
                #     _, Hc, Wc = rendering.shape
                #     crop_before = rendering_before[:, :Hc, :Wc]  # crop tho, chi de debug
                #     diff = (rendering - crop_before).abs()
                #     print(f"[DEBUG] redistort diff: mean={diff.mean().item():.6f} "
                #         f"max={diff.max().item():.6f}")
                #     torchvision.utils.save_image(rendering_before, "/kaggle/working/debug_before_redistort.png")
                #     torchvision.utils.save_image(rendering, "/kaggle/working/debug_after_redistort.png")

            out_path = scene_dir / row["image_name"]
            torchvision.utils.save_image(rendering, out_path)
            del camera, rendering

    print(f"Rendered {len(rows)} images for {scene_name} from iteration {loaded_iter} -> {scene_dir}")

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
    parser.add_argument("--supersample_factor", default=1.0, type=float, help="Scale factor for supersampling (e.g. 2.0)")

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
        supersample_factor=args.supersample_factor
    )