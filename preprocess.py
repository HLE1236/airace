import argparse
import os
import shutil
import subprocess
import tempfile
import concurrent.futures
from pathlib import Path

import numpy as np
from PIL import Image

from utils.read_write_model import (
    read_images_binary,
    read_cameras_binary,
    read_points3D_binary,
    write_images_binary,
    write_cameras_binary,
    write_points3D_binary,
)

def preprocess_scene(scene_path):

    scene_path = Path(scene_path)

    images_dir = scene_path / "train" / "images"
    sparse_path = scene_path / "train" / "sparse" / "0"

    cameras = read_cameras_binary(str(sparse_path / "cameras.bin"))
    images = read_images_binary(str(sparse_path / "images.bin"))
    points3D = read_points3D_binary(str(sparse_path / "points3D.bin"))

    existing_files = set(os.listdir(images_dir))

    # dùng set để tra cứu O(1)
    missing_ids = {
        img_id
        for img_id, img in images.items()
        if img.name not in existing_files
    }

    print(f"scene: {scene_path.name}")
    print(f"Tổng ảnh: {len(images)}")
    print(f"Số ảnh missing: {len(missing_ids)}")

    for img_id in missing_ids:
        del images[img_id]

    print(f"Còn lại sau khi xóa: {len(images)}")
    # ghi đè luôn
    write_cameras_binary(cameras, str(sparse_path / "cameras.bin"))
    write_images_binary(images, str(sparse_path / "images.bin"))
    write_points3D_binary(points3D, str(sparse_path / "points3D.bin"))

    print(f"Đã cập nhật reconstruction.")


def _run_colmap_undistorter(image_path, input_sparse_path, output_path,
                             blank_pixels, min_scale, max_scale):
    subprocess.run(
        [
            "colmap", "image_undistorter",
            "--image_path", str(image_path),
            "--input_path", str(input_sparse_path),
            "--output_path", str(output_path),
            "--output_type", "COLMAP",
            "--blank_pixels", str(blank_pixels),
            "--min_scale", str(min_scale),
            "--max_scale", str(max_scale),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def _sync_images_bin_names_to_disk(sparse_dir, image_dir):
    """
    Sau khi embed alpha co the doi duoi anh (vd .jpg -> .png). Ham nay cap nhat lai
    truong 'name' trong images.bin cho khop voi ten file thuc te tren dia (chi doi
    phan duoi, giu nguyen stem), roi ghi de lai file.
    """
    sparse_dir = Path(sparse_dir)
    image_dir = Path(image_dir)

    images = read_images_binary(str(sparse_dir / "images.bin"))
    on_disk_stems = {Path(f).stem: f for f in os.listdir(image_dir)}

    changed = 0
    for img_id, img in list(images.items()):
        stem = Path(img.name).stem
        if stem in on_disk_stems and on_disk_stems[stem] != img.name:
            images[img_id] = img._replace(name=on_disk_stems[stem])
            changed += 1
        elif stem not in on_disk_stems:
            print(f"  WARNING: khong tim thay file tren dia cho '{img.name}' (stem={stem})")

    if changed:
        write_images_binary(images, str(sparse_dir / "images.bin"))
        print(f"  Đã đồng bộ tên {changed} ảnh trong images.bin (đổi đuôi sau embed alpha)")


def generate_alpha_masks_and_embed_from_original(orig_image_dir, orig_sparse_dir, undistorted_image_dir,
                                                   blank_pixels=1.0, min_scale=1.0, max_scale=2.0,
                                                   threshold=127):
    """
    Goi SAU khi undistort_scene() da chay xong (undistorted_image_dir la anh da undistort
    that, PNG/JPG thuong 3-kenh), NHUNG truoc do orig_image_dir/orig_sparse_dir (SIMPLE_RADIAL
    goc) phai con giu nguyen -- vi vay ham nay nhan truc tiep duong dan toi ban COPY cua
    anh/sparse goc (xem preprocess_dataset() de biet cach giu ban copy nay).

    1. Warp anh trang qua colmap image_undistorter voi CUNG tham so nhu lan undistort that,
       dung sparse model GOC -> ra vung hop le (mask).
    2. Doc anh da undistort that (RGB) + mask tuong ung, ghep thanh RGBA, GHI DE lai
       thanh .png (bat buoc .png vi .jpg khong ho tro kenh alpha) trong undistorted_image_dir.
    """
    orig_image_dir = Path(orig_image_dir)
    orig_sparse_dir = Path(orig_sparse_dir)
    undistorted_image_dir = Path(undistorted_image_dir)

    white_dir = Path(tempfile.mkdtemp(prefix="white_src_", dir=undistorted_image_dir.parent))
    mask_out_dir = Path(tempfile.mkdtemp(prefix="white_undist_", dir=undistorted_image_dir.parent))

    try:
        image_names = sorted(os.listdir(orig_image_dir))
        
        def create_white_image(name):
            with Image.open(orig_image_dir / name) as im:
                w, h = im.size
            Image.new("RGB", (w, h), (255, 255, 255)).save(white_dir / name)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            list(executor.map(create_white_image, image_names))

        print(f"Generating alpha masks (COLMAP CLI on white images)...")
        _run_colmap_undistorter(
            white_dir, orig_sparse_dir, mask_out_dir,
            blank_pixels=blank_pixels, min_scale=min_scale, max_scale=max_scale,
        )
        mask_src_dir = mask_out_dir / "images"

        mask_names = os.listdir(mask_src_dir)
        def merge_mask(name):
            mask_path = mask_src_dir / name
            undist_img_path = undistorted_image_dir / name
            if not undist_img_path.exists():
                print(f"  WARNING: khong tim thay anh undistort tuong ung cho mask {name}, bo qua")
                return 0

            mask_im = np.array(Image.open(mask_path).convert("L"))
            binary_alpha = (mask_im > threshold).astype(np.uint8) * 255

            rgb_im = Image.open(undist_img_path).convert("RGB")
            rgba = np.dstack([np.array(rgb_im), binary_alpha])

            out_path = undist_img_path.with_suffix(".png")
            Image.fromarray(rgba, mode="RGBA").save(out_path)
            if out_path != undist_img_path:
                undist_img_path.unlink()  # xoa file .jpg cu neu doi sang .png
            return 1

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(merge_mask, mask_names))
        embedded = sum(results)

        print(f"Đã ghi alpha channel vào {embedded} ảnh undistort tại {undistorted_image_dir}")

    except subprocess.CalledProcessError as e:
        print(f"LOI khi sinh alpha mask:")
        print(e.stderr)
        raise
    finally:
        shutil.rmtree(white_dir, ignore_errors=True)
        shutil.rmtree(mask_out_dir, ignore_errors=True)


def undistort_scene(scene_path, blank_pixels=1.0, min_scale=1.0, max_scale=2.0, embed_alpha_mask=True):
    """
    Undistort dung COLMAP CLI (C++) - nhanh hon nhieu so voi pycolmap python API,
    vi chay truc tiep binary C++, khong qua overhead Python/GIL.

    blank_pixels=1.0: chap nhan toi da pixel den, uu tien KHONG CROP mat noi dung goc
                       (tuong duong --blank_pixels 1 trong docs COLMAP).
    min_scale/max_scale: khoang scale COLMAP duoc phep tu dieu chinh de thoa blank_pixels.

    LUU Y: COLMAP tu dong chon scale NHO NHAT du de thoa dieu kien blank_pixels,
    khong co nghia la no se dung het max_scale. Neu can canvas lon hon, phai
    tu can thiep them (vd dung script CV2 thu cong da test o cho khac).

    embed_alpha_mask=True: sau khi undistort, ghi vung fill den (do COLMAP tao ra khi
    mo rong canvas) vao KENH ALPHA cua chinh anh, luu thanh PNG RGBA. Day la convention
    ma scene/cameras.py (graphdeco-inria) doc: anh 4-kenh -> kenh 4 la alpha_mask tu dong
    nhan vao loss; anh 3-kenh -> khong mask gi ca (viem den se bi tinh la pixel that ->
    loss sai, Gaussian sinh ra de "giai thich" vung den).
    """
    scene_path = Path(scene_path)

    image_dir = scene_path / "train" / "images"
    sparse_dir = scene_path / "train" / "sparse" / "0"

    tmp_dir = Path(tempfile.mkdtemp(prefix="undistort_", dir=scene_path.parent))
    orig_backup_dir = None

    try:
        if embed_alpha_mask:
            # Giu ban copy anh + sparse GOC (SIMPLE_RADIAL) truoc khi bi ghi de,
            # vi buoc sinh mask can warp field dung CHINH sparse model goc nay.
            orig_backup_dir = Path(tempfile.mkdtemp(prefix="orig_backup_", dir=scene_path.parent))
            shutil.copytree(image_dir, orig_backup_dir / "images")
            shutil.copytree(sparse_dir, orig_backup_dir / "sparse0")

        print(f"Undistorting {scene_path.name} (COLMAP CLI)...")
        _run_colmap_undistorter(
            image_dir, sparse_dir, tmp_dir,
            blank_pixels=blank_pixels, min_scale=min_scale, max_scale=max_scale,
        )

        # Thay ảnh
        shutil.rmtree(image_dir)
        shutil.copytree(tmp_dir / "images", image_dir)

        # Thay sparse
        shutil.rmtree(sparse_dir)
        shutil.copytree(tmp_dir / "sparse", sparse_dir)

        if embed_alpha_mask:
            generate_alpha_masks_and_embed_from_original(
                orig_image_dir=orig_backup_dir / "images",
                orig_sparse_dir=orig_backup_dir / "sparse0",
                undistorted_image_dir=image_dir,
                blank_pixels=blank_pixels, min_scale=min_scale, max_scale=max_scale,
            )
            # QUAN TRONG: neu anh goc khong phai .png (vd .jpg), buoc embed da doi
            # duoi thanh .png (bat buoc, vi jpg khong ho tro kenh alpha). images.bin
            # van con luu ten file CU (.jpg) -> phai dong bo lai, neu khong
            # preprocess_scene() se tuong toan bo anh la "missing" va xoa sach.
            _sync_images_bin_names_to_disk(sparse_dir, image_dir)

    except subprocess.CalledProcessError as e:
        print(f"LOI khi undistort {scene_path.name}:")
        print(e.stderr)
        raise
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        if orig_backup_dir is not None:
            shutil.rmtree(orig_backup_dir, ignore_errors=True)

def dense_reconstruction(scene_path, gpu_index=0):
    """Run COLMAP PatchMatch stereo + fusion to produce dense.ply"""
    scene_path = Path(scene_path)
    image_dir = scene_path / "train" / "images"
    sparse_dir = scene_path / "train" / "sparse" / "0"
    dense_dir = scene_path / "train" / "dense"
    
    dense_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating dense workspace for {scene_path.name}...")
    subprocess.run([
        "colmap", "image_undistorter",
        "--image_path", str(image_dir),
        "--input_path", str(sparse_dir),
        "--output_path", str(dense_dir),
        "--output_type", "COLMAP",
    ], check=True)
    
    print(f"Running PatchMatch Stereo for {scene_path.name}...")
    subprocess.run([
        "colmap", "patch_match_stereo",
        "--workspace_path", str(dense_dir),
        "--workspace_format", "COLMAP",
        "--PatchMatchStereo.geom_consistency", "true",
        "--PatchMatchStereo.gpu_index", str(gpu_index),
        "--PatchMatchStereo.max_image_size", "2000",
        "--PatchMatchStereo.window_radius", "5",
        "--PatchMatchStereo.filter_min_ncc", "0.1",
        "--PatchMatchStereo.num_iterations", "5",
    ], check=True)
    
    dense_ply = dense_dir / "fused.ply"
    print(f"Running Stereo Fusion for {scene_path.name}...")
    subprocess.run([
        "colmap", "stereo_fusion",
        "--workspace_path", str(dense_dir),
        "--workspace_format", "COLMAP",
        "--output_path", str(dense_ply),
        "--StereoFusion.min_num_pixels", "3",
        "--StereoFusion.max_reproj_error", "2",
    ], check=True)
    
    stereo_folder = dense_dir / "stereo"
    if stereo_folder.exists():
        print(f"Cleaning up heavy intermediate stereo files at {stereo_folder}...")
        shutil.rmtree(stereo_folder, ignore_errors=True)
        
    return dense_ply

def preprocess_dataset(path, output_dir, blank_pixels=1.0, min_scale=1.0, max_scale=2.0, subset=[], enable_dense=False):
    path = Path(path)
    output_dir = Path(output_dir)

    shutil.copytree(path, output_dir, dirs_exist_ok=True)

    for scene in os.listdir(path):

        if len(subset) == 0 or scene in subset:
            output_scene_path = output_dir / scene
            undistort_scene(
                output_scene_path,
                blank_pixels=blank_pixels, min_scale=min_scale, max_scale=max_scale,
                embed_alpha_mask=True,
            )
            preprocess_scene(output_scene_path)
            
            if enable_dense:
                dense_reconstruction(output_scene_path)

def validate(path):
    path = Path(path)
    for scene in os.listdir(path):
        scene_path = path / scene
        images_dir = scene_path / "train" / "images"
        sparse_path = scene_path / "train" / "sparse" / "0"

        cameras = read_cameras_binary(str(sparse_path / "cameras.bin"))
        images = read_images_binary(str(sparse_path / "images.bin"))
        points3D = read_points3D_binary(str(sparse_path / "points3D.bin"))

        on_disk = set(os.listdir(images_dir))
        registered = {img.name for img in images.values()}

        missing = registered - on_disk
        extra = on_disk - registered
        non_pinhole = [c for c in cameras.values() if c.model != "PINHOLE"]

        non_rgba = []
        for name in list(on_disk)[:50]:  # sample de khong mo het anh
            try:
                with Image.open(images_dir / name) as im:
                    if im.mode != "RGBA":
                        non_rgba.append(name)
            except Exception:
                pass

        print(f"scene: {scene}")
        print(f"  images (sparse/disk): {len(images)}/{len(on_disk)}")
        print(f"  points3D: {len(points3D)}")
        if missing:
            print(f"  ⚠ missing on disk: {len(missing)} e.g. {list(missing)[:3]}")
        if extra:
            print(f"  ⚠ extra on disk (not in sparse): {len(extra)}")
        if non_pinhole:
            print(f"  ⚠ non-PINHOLE cameras (undistort may have failed): {len(non_pinhole)}")
        if non_rgba:
            print(f"  ⚠ images khong phai RGBA (thieu alpha mask), sample: {non_rgba[:3]}")
        if not missing and not extra and not non_pinhole and not non_rgba:
            print("  ✓ OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess COLMAP dataset by removing missing images and updating sparse model."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input dataset directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="/kaggle/working/cleaned_inputs"
    )

    parser.add_argument(
        "--subset",
        nargs="+",
        type=str,
        default=[]
    )
    parser.add_argument(
        "--enable_dense",
        action="store_true",
        help="Run COLMAP PatchMatch and Stereo Fusion to generate dense point cloud."
    )

    args = parser.parse_args()

    preprocess_dataset(args.input, args.output, subset=args.subset, enable_dense=args.enable_dense)
    # validate(args.output)

# preprocess_dataset(r"C:\contest\VAR2026\phase1\private_set1", r"C:\contest\VAR2026\dataset")