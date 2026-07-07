import argparse
import os
import shutil
from pathlib import Path

from utils.read_write_model import (
    read_images_binary,
    read_cameras_binary,
    read_points3D_binary,
    write_images_binary,
    write_cameras_binary,
    write_points3D_binary,
)

def preprocess_scene(path, output_dir):

    path = Path(path)
    output_dir = Path(output_dir)

    images_dir = path / "train" / "images"
    sparse_path = path / "train" / "sparse" / "0"

    cameras = read_cameras_binary(str(sparse_path / "cameras.bin"))
    images = read_images_binary(str(sparse_path / "images.bin"))
    points3D = read_points3D_binary(str(sparse_path / "points3D.bin"))

    # images that actualy exsist in images dir
    existing_files = set(os.listdir(images_dir))

    # extrinsics that do not have exsisting images
    missing_ids = [img_id for img_id, img in images.items() if img.name not in existing_files]

    print(f"scene: {path.name}")
    print(f"Tổng ảnh: {len(images)}")
    print(f"Số ảnh missing: {len(missing_ids)}")

    # Xóa hẳn khỏi dict images
    for img_id in missing_ids:
        del images[img_id]

    print(f"Còn lại sau khi xóa: {len(images)}")

    # Dọn point3D: xóa các observation trỏ tới ảnh đã xóa, xóa point nếu track rỗng
    removed_points = 0
    for pt_id in list(points3D.keys()):
        pt = points3D[pt_id]

        # delete observations of images that is not exsist 
        keep_mask = [img_id not in missing_ids for img_id in pt.image_ids]

        # delete if none of the observation exsist
        if not any(keep_mask):
            del points3D[pt_id]
            removed_points += 1
        elif not all(keep_mask):
            new_image_ids = pt.image_ids[keep_mask]
            new_point2D_idxs = pt.point2D_idxs[keep_mask]
            points3D[pt_id] = pt._replace(image_ids=new_image_ids, point2D_idxs=new_point2D_idxs)

    print(f"Đã xóa {removed_points} points3D không còn observation nào")
    print(f"Points3D còn lại: {len(points3D)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_cameras_binary(cameras, str(output_dir / "cameras.bin"))
    write_images_binary(images, str(output_dir / "images.bin"))
    write_points3D_binary(points3D, str(output_dir / "points3D.bin"))
    print(f"Đã ghi kết quả tại: {output_dir}")


def preprocess_dataset(path, output_dir):
    path = Path(path)
    output_dir = Path(output_dir)

    shutil.copytree(path, output_dir, dirs_exist_ok=True)

    for scene in os.listdir(path):
        scene_path = path / scene
        ouput_scene_path = output_dir / scene / "train" / "sparse" / "0"
        preprocess_scene(scene_path, ouput_scene_path)        


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
        required=True,
        help="Output dataset directory",
    )

    args = parser.parse_args()

    preprocess_dataset(args.input, args.output)

# preprocess_dataset(r"C:\contest\VAR2026\phase1\private_set1", r"C:\contest\VAR2026\dataset")