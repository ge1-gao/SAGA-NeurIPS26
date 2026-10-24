"""MPEG MIV CTC texture sequence (perspective only: <seq>.json + vN_texture_WxH_yuv420p10le.yuv) -> the Neu3D layout:
png/camXX/NNNNN.png (1-indexed, view vN -> camNN) + colmap_0/manual (PINHOLE per view, OpenCV world-to-camera).

Cameras follow the OMAF convention of the MIV JSON: x forward, y left, z up; Rotation = (yaw, pitch, roll) in degrees,
camera-to-world R = Rz(yaw) Ry(pitch) Rx(roll). Colour: 10-bit limited-range BT.709 YCbCr -> 8-bit RGB.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import cv2
import numpy as np
import tyro

from saga.data.prep_neu3d import rotmat_to_qvec

OMAF_FROM_CV = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float64)  # columns: cv x, y, z in OMAF


@dataclass
class Args:
    seq_dir: str  # MIV_texture_65f/<seq id>, e.g. D01
    out_dir: str
    n_frames: int = 65
    workers: int = 16


def euler_to_rotmat(yaw: float, pitch: float, roll: float) -> np.ndarray:
    y, p, r = np.radians([yaw, pitch, roll])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    return Rz @ Ry @ Rx


def yuv420p10_to_rgb(buf: np.ndarray, w: int, h: int) -> np.ndarray:
    y = buf[:w * h].reshape(h, w).astype(np.float32)
    u = buf[w * h:w * h * 5 // 4].reshape(h // 2, w // 2).astype(np.float32)
    v = buf[w * h * 5 // 4:].reshape(h // 2, w // 2).astype(np.float32)
    u, v = (cv2.resize(c, (w, h), interpolation=cv2.INTER_LINEAR) for c in (u, v))
    y, cb, cr = (y - 64) / 876, (u - 512) / 896, (v - 512) / 896
    rgb = np.stack([y + 1.5748 * cr, y - 0.1873 * cb - 0.4681 * cr, y + 1.8556 * cb], -1)
    return (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)


def convert_view(yuv: str, out_dir: str, w: int, h: int, n_frames: int) -> None:
    os.makedirs(out_dir, exist_ok=True)
    n = w * h * 3 // 2
    with open(yuv, "rb") as f:
        for t in range(n_frames):
            dst = os.path.join(out_dir, f"{t + 1:05d}.png")
            buf = np.frombuffer(f.read(n * 2), dtype="<u2")
            assert len(buf) == n, f"{yuv}: frame {t} truncated"
            if not os.path.exists(dst):
                cv2.imwrite(dst, yuv420p10_to_rgb(buf, w, h)[..., ::-1])


def main(a: Args) -> None:
    (meta,) = [f for f in os.listdir(a.seq_dir) if f.endswith(".json")]
    with open(os.path.join(a.seq_dir, meta)) as f:
        views = [c for c in json.load(f)["cameras"] if c["Name"] != "viewport"]
    assert all(c["Projection"] == "Perspective" for c in views), "ERP sequences are not supported"

    out = os.path.join(a.out_dir, "colmap_0", "manual")
    os.makedirs(out, exist_ok=True)
    jobs = []
    with open(os.path.join(out, "cameras.txt"), "w") as fc, open(os.path.join(out, "images.txt"), "w") as fi:
        for i, c in enumerate(sorted(views, key=lambda c: int(c["Name"][1:])), 1):
            name = f"cam{int(c['Name'][1:]):02d}"
            w, h = c["Resolution"]
            (fx, fy), (cx, cy) = c["Focal"], c["Principle_point"]
            c2w = np.eye(4)
            c2w[:3, :3] = euler_to_rotmat(*c["Rotation"]) @ OMAF_FROM_CV
            c2w[:3, 3] = c["Position"]
            w2c = np.linalg.inv(c2w)
            fc.write(f"{i} PINHOLE {w} {h} {fx} {fy} {cx} {cy}\n")
            q, t = rotmat_to_qvec(w2c[:3, :3]), w2c[:3, 3]
            fi.write(f"{i} {' '.join(map(str, q))} {' '.join(map(str, t))} {i} {name}.png\n\n")
            yuv = os.path.join(a.seq_dir, f"{c['Name']}_texture_{w}x{h}_yuv420p10le.yuv")
            jobs.append((yuv, os.path.join(a.out_dir, "png", name), w, h, a.n_frames))
    open(os.path.join(out, "points3D.txt"), "w").close()
    with ProcessPoolExecutor(a.workers) as ex:
        list(ex.map(convert_view, *zip(*jobs)))
    print(f"{len(views)} views x {a.n_frames} frames -> {a.out_dir}")


if __name__ == "__main__":
    main(tyro.cli(Args))
