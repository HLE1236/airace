import os
import argparse
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Undistort a COLMAP dataset for 3DGS training.")
    parser.add_argument("--source_path", "-s", required=True, type=str, help="Path to the dataset (containing images/ and sparse/0/)")
    parser.add_argument("--output_path", "-o", type=str, default="", help="Path to save undistorted dataset (defaults to source_path + '_undistorted')")
    parser.add_argument("--colmap_executable", default="colmap", type=str)
    args = parser.parse_args()

    source_path = Path(args.source_path)
    if args.output_path == "":
        output_path = source_path.parent / (source_path.name + "_undistorted")
    else:
        output_path = Path(args.output_path)

    images_path = source_path / "images"
    sparse_path = source_path / "sparse" / "0"

    if not images_path.exists() or not sparse_path.exists():
        print(f"Error: {source_path} must contain 'images' and 'sparse/0' directories.")
        return

    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Undistorting images from {source_path} into {output_path}...")
    print(f"This will correct any SIMPLE_RADIAL distortion (like k1) and convert to PINHOLE.")
    
    cmd = [
        args.colmap_executable, "image_undistorter",
        "--image_path", str(images_path),
        "--input_path", str(sparse_path),
        "--output_path", str(output_path),
        "--output_type", "COLMAP"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("\nUndistortion complete!")
        print(f"Your undistorted dataset is at: {output_path}")
        print("You can now train 3DGS on this undistorted dataset.")
    except subprocess.CalledProcessError as e:
        print(f"\nError running COLMAP: {e}")
        print("Please ensure COLMAP is installed and accessible in your PATH.")

if __name__ == "__main__":
    main()
