"""Export one GOP to the 4D Scaffold GS N3DV format (same frames, cameras, test view and SfM points).
Train it with --time_duration 0.0 <gop_len / fps>."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
from plyfile import PlyData, PlyElement
import tyro

from saga.data.neu3d import load_points, read_cameras


@dataclass
class Args:
    scene_dir: str
    points_dir: str
    out_dir: str
    gop_start: int = 0
    gop_len: int = 50
    test_cams: tuple[str, ...] = ("cam00",)
    fps: float = 30.0


def main(a: Args) -> None:
    cams = read_cameras(os.path.join(a.scene_dir, "colmap_0", "manual"))
    names = sorted(n for n in cams if os.path.isdir(os.path.join(a.scene_dir, "png", n)))
    os.makedirs(os.path.join(a.out_dir, "images"), exist_ok=True)
    splits = {"train": [], "test": []}
    for n in names:
        c = cams[n]
        c2w = np.linalg.inv(c.viewmat)
        c2w[:3, 1:3] *= -1  # OpenCV -> OpenGL
        for t in range(a.gop_len):
            stem = f"{n}_{t:04d}"
            dst = os.path.join(a.out_dir, "images", stem + ".png")
            if not os.path.lexists(dst):
                os.symlink(os.path.abspath(os.path.join(a.scene_dir, "png", n, f"{a.gop_start + t + 1:05d}.png")), dst)
            frame = {"file_path": f"images/{stem}", "time": t / a.fps, "transform_matrix": c2w.tolist(),
                     "fl_x": c.K[0, 0], "fl_y": c.K[1, 1], "cx": c.K[0, 2], "cy": c.K[1, 2], "w": c.width, "h": c.height}
            if "all" in a.test_cams:  # MIV CTC: every source view is trained on and evaluated
                splits["train"].append(frame)
                splits["test"].append(frame)
            else:
                splits["test" if n in a.test_cams else "train"].append(frame)
    for k, frames in splits.items():
        with open(os.path.join(a.out_dir, f"transforms_{k}.json"), "w") as f:
            json.dump({"frames": frames}, f)

    pts = load_points(a.points_dir, a.gop_start, a.gop_len)
    xyz, rgb = pts["xyz"], np.clip(pts["rgb"] * 255, 0, 255).astype(np.uint8)
    v = np.empty(len(xyz), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
                                  ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    for i, k in enumerate("xyz"):
        v[k] = xyz[:, i]
    for k in ("nx", "ny", "nz"):
        v[k] = 0
    for i, k in enumerate(("red", "green", "blue")):
        v[k] = rgb[:, i]
    PlyData([PlyElement.describe(v, "vertex")]).write(os.path.join(a.out_dir, "points3d.ply"))
    print(f"train {len(splits['train'])} / test {len(splits['test'])} images, {len(xyz)} points; "
          f"use --time_duration 0.0 {a.gop_len / a.fps:.4f}")


if __name__ == "__main__":
    main(tyro.cli(Args))
