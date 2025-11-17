#!/usr/bin/env python3
"""Convert CAMUS 4CH cine NIfTI volumes into MP4 clips for EchoFM pretraining."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import imageio.v2 as imageio
import nibabel as nib
import numpy as np


def _normalize_uint8(volume: np.ndarray) -> np.ndarray:
    """Normalize arbitrary float/int volume to uint8 0-255."""
    vol = volume.astype(np.float32)
    vol -= vol.min()
    peak = vol.max()
    if peak > 0:
        vol /= peak
    return (vol * 255.0).clip(0, 255).astype(np.uint8)


def convert_case(nifti_path: Path, output_dir: Path, fps: int) -> Path:
    nifti = nib.load(str(nifti_path))
    volume = nifti.get_fdata()  # H x W x T
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D cine volume in {nifti_path}, got shape {volume.shape}")

    frames = np.transpose(volume, (2, 0, 1))  # T, H, W
    frames = _normalize_uint8(frames)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{nifti_path.stem}.mp4"
    with imageio.get_writer(out_path, fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)
    return out_path


def iter_camus_files(root: Path, pattern: str) -> Iterable[Path]:
    for patient_dir in sorted(root.glob("patient*")):
        yield from patient_dir.glob(pattern)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_root",
        required=True,
        type=Path,
        help="Path to CAMUS database_nifti directory",
    )
    parser.add_argument(
        "--pattern",
        default="*_4CH_half_sequence.nii.gz",
        help="Glob pattern inside each patient directory (default: *_4CH_half_sequence.nii.gz)",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=Path,
        help="Where to write the generated MP4 clips",
    )
    parser.add_argument("--fps", type=int, default=25, help="Output video FPS (default: 25)")
    args = parser.parse_args()

    if not args.input_root.exists():
        raise FileNotFoundError(f"Input root {args.input_root} does not exist")

    mp4_dir = args.output_dir
    mp4_dir.mkdir(parents=True, exist_ok=True)

    nifti_paths = list(iter_camus_files(args.input_root, args.pattern))
    if not nifti_paths:
        raise RuntimeError("No CAMUS cines found with the given pattern. Double-check the paths.")

    print(f"Discovered {len(nifti_paths)} NIfTI volumes. Converting to {mp4_dir} ...")
    for nii in nifti_paths:
        out = convert_case(nii, mp4_dir, args.fps)
        print(f"Wrote {out}")

    print("Done.")


if __name__ == "__main__":
    main()
