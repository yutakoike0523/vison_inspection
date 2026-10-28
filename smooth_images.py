"""Smooth inspection images with a Gaussian filter.

Run:
    python smooth_images.py

The default output directory is recreated on every run, so the results always
correspond to the current input and settings. Source images are never changed.
"""
import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / 'data20260903' / 'data20260903'
EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def read_image(path: Path) -> np.ndarray:
    """Read an image while preserving its original channel count and bit depth."""
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f'Cannot read image: {path}')
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    success, encoded = cv2.imencode(path.suffix, image)
    if not success:
        raise ValueError(f'Cannot encode image: {path.name}')
    path.write_bytes(encoded.tobytes())


def refresh_output_directory(output_dir: Path, input_dir: Path) -> None:
    """Delete only the previous generated output, never the source directory."""
    output_dir = output_dir.resolve()
    input_dir = input_dir.resolve()
    if output_dir == input_dir or output_dir in input_dir.parents:
        raise ValueError('Output directory must not be the input directory or one of its parents.')
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT,
                        help='Folder containing source images.')
    parser.add_argument('--output', type=Path, default=DEFAULT_INPUT / 'smoothed',
                        help='Generated folder, replaced on every run.')
    parser.add_argument('--kernel', type=int, default=3,
                        help='Odd Gaussian kernel size in pixels (default: 3).')
    parser.add_argument('--sigma', type=float, default=0.0,
                        help='Gaussian sigma; 0 lets OpenCV choose it from --kernel.')
    parser.add_argument('--recursive', action='store_true',
                        help='Also process image files in subfolders.')
    args = parser.parse_args()

    if args.kernel < 3 or args.kernel % 2 == 0:
        parser.error('--kernel must be an odd integer of at least 3.')
    if args.sigma < 0:
        parser.error('--sigma must be zero or positive.')

    input_dir = args.input.resolve()
    if not input_dir.is_dir():
        parser.error(f'Input folder does not exist: {input_dir}')
    output_dir = args.output.resolve()
    iterator = input_dir.rglob('*') if args.recursive else input_dir.iterdir()
    sources = sorted(path for path in iterator
                     if path.is_file() and path.suffix.lower() in EXTENSIONS
                     and output_dir not in path.parents)
    if not sources:
        parser.error('No supported image files found.')

    refresh_output_directory(output_dir, input_dir)
    for source in sources:
        relative_path = source.relative_to(input_dir)
        destination = output_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        smoothed = cv2.GaussianBlur(read_image(source), (args.kernel, args.kernel), args.sigma)
        write_image(destination, smoothed)
        print(f'{relative_path} -> {destination.relative_to(output_dir)}')
    print(f'Completed: {len(sources)} images -> {output_dir}')


if __name__ == '__main__':
    main()
