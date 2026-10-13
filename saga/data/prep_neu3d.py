"""Raw Neu3D (camXX.mp4 + LLFF poses_bounds.npy) -> png/camXX/NNNNN.png + colmap_0/manual (OpenCV, world-to-camera)."""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import cv2
import numpy as np
import tyro


@dataclass
class Args:
    scene_dir: str
    n_frames: int = 50
    workers: int = 21


def rotmat_to_qvec(R: np.ndarray) -> np.ndarray:
    w = np.sqrt(max(1e-12, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    q = np.array([w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)])
    return q / np.linalg.norm(q)


def write_manual(scene_dir: str, names: list[str]) -> None:
    arr = np.load(os.path.join(scene_dir, "poses_bounds.npy"))
    poses = arr[:, :15].reshape(-1, 3, 5)
    assert len(poses) == len(names), (len(poses), len(names))
    out = os.path.join(scene_dir, "colmap_0", "manual")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "cameras.txt"), "w") as fc, open(os.path.join(out, "images.txt"), "w") as fi:
        for i, (name, p) in enumerate(zip(names, poses), 1):
            h, w, f = p[:, 4]
            c2w = np.eye(4)
            c2w[:3, :3] = np.stack([p[:, 1], p[:, 0], -p[:, 2]], 1)  # LLFF -> OpenCV
            c2w[:3, 3] = p[:, 3]
            w2c = np.linalg.inv(c2w)
            fc.write(f"{i} PINHOLE {int(w)} {int(h)} {f} {f} {w / 2} {h / 2}\n")
            q, t = rotmat_to_qvec(w2c[:3, :3]), w2c[:3, 3]
            fi.write(f"{i} {' '.join(map(str, q))} {' '.join(map(str, t))} {i} {name}.png\n\n")
    open(os.path.join(out, "points3D.txt"), "w").close()


def extract(video: str, out_dir: str, n_frames: int) -> int:
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video)
    n = 0
    while n < n_frames:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(os.path.join(out_dir, f"{n + 1:05d}.png"), frame)
        n += 1
    return n


def main(a: Args) -> None:
    names = sorted(f[:-4] for f in os.listdir(a.scene_dir) if f.startswith("cam") and f.endswith(".mp4"))
    write_manual(a.scene_dir, names)
    with ProcessPoolExecutor(a.workers) as ex:
        counts = ex.map(extract, [os.path.join(a.scene_dir, f"{n}.mp4") for n in names],
                        [os.path.join(a.scene_dir, "png", n) for n in names], [a.n_frames] * len(names))
        for n, c in zip(names, counts):
            print(f"{n}: {c} frames")


if __name__ == "__main__":
    main(tyro.cli(Args))
