"""Neu3D GOP loader: cameras from colmap_0/manual, frames from png/camXX/NNNNN.png (1-indexed)."""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image


def qvec_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
    ])


@dataclass
class Camera:
    name: str
    K: np.ndarray  # [3, 3] at full resolution
    viewmat: np.ndarray  # [4, 4] world-to-camera
    width: int
    height: int


def read_cameras(manual_dir: str) -> dict[str, Camera]:
    intr = {}
    with open(os.path.join(manual_dir, "cameras.txt")) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            el = line.split()
            assert el[1] == "PINHOLE", el[1]
            w, h, fx, fy, cx, cy = map(float, el[2:8])
            intr[int(el[0])] = (int(w), int(h), np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]))
    cams = {}
    with open(os.path.join(manual_dir, "images.txt")) as f:
        for line in f:
            el = line.split()
            if len(el) < 10 or line.startswith("#"):
                continue
            q = np.array(list(map(float, el[1:5])))
            t = np.array(list(map(float, el[5:8])))
            w, h, K = intr[int(el[8])]
            vm = np.eye(4)
            vm[:3, :3] = qvec_to_rotmat(q)
            vm[:3, 3] = t
            name = os.path.splitext(el[9])[0]
            cams[name] = Camera(name, K, vm, w, h)
    return cams


def _load_image(path: str, size: tuple[int, int]) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    if im.size != size:
        im = im.resize(size, Image.BICUBIC)
    return np.asarray(im)


class Neu3DGOP:
    """All views of frames [start, start+length) held as a uint8 tensor on `device`."""

    def __init__(self, scene_dir: str, start: int, length: int, downscale: int = 2,
                 test_cams: tuple[str, ...] = ("cam00",), device: str = "cuda", workers: int = 16):
        self.scene_dir, self.start, self.length = scene_dir, start, length
        all_cams = read_cameras(os.path.join(scene_dir, "colmap_0", "manual"))
        names = sorted(n for n in all_cams if os.path.isdir(os.path.join(scene_dir, "png", n)))
        self.cams = [all_cams[n] for n in names]
        self.test_idx = [i for i, c in enumerate(self.cams) if c.name in test_cams]
        self.train_idx = [i for i, c in enumerate(self.cams) if c.name not in test_cams]
        assert self.test_idx and self.train_idx

        c0 = self.cams[0]
        self.width, self.height = c0.width // downscale, c0.height // downscale
        sx, sy = self.width / c0.width, self.height / c0.height
        Ks = []
        for c in self.cams:
            K = c.K.copy()
            K[0] *= sx
            K[1] *= sy
            Ks.append(K)
        self.Ks = torch.tensor(np.stack(Ks), dtype=torch.float32, device=device)
        self.viewmats = torch.tensor(np.stack([c.viewmat for c in self.cams]), dtype=torch.float32, device=device)
        self.camtoworlds = torch.linalg.inv(self.viewmats)

        paths = [os.path.join(scene_dir, "png", c.name, f"{start + t + 1:05d}.png")
                 for c in self.cams for t in range(length)]
        with ProcessPoolExecutor(workers) as ex:
            ims = list(ex.map(partial(_load_image, size=(self.width, self.height)), paths, chunksize=8))
        self.images = torch.from_numpy(np.stack(ims)).view(
            len(self.cams), length, self.height, self.width, 3).to(device)
        self.device = device

    def time_of(self, t: int) -> float:
        """τ ∈ [0, 1)."""
        return t / self.length

    @property
    def scene_scale(self) -> float:
        centers = self.camtoworlds[:, :3, 3]
        return float((centers - centers.mean(0)).norm(dim=-1).max() * 1.1)

    def sample(self, n_views: int, generator: torch.Generator | None = None) -> dict:
        """One random frame from `n_views` random training cameras."""
        t = int(torch.randint(self.length, (1,), generator=generator))
        perm = torch.randperm(len(self.train_idx), generator=generator)[:n_views]
        idx = torch.tensor([self.train_idx[i] for i in perm], device=self.device)
        return self.batch(idx, t)

    def batch(self, cam_idx: torch.Tensor, t: int) -> dict:
        return {
            "image": self.images[cam_idx, t].float() / 255.0,  # [C, H, W, 3]
            "cam": cam_idx,
            "Ks": self.Ks[cam_idx],
            "viewmats": self.viewmats[cam_idx],
            "camtoworlds": self.camtoworlds[cam_idx],
            "t": t,
            "tau": self.time_of(t),
        }


def load_points(points_dir: str, start: int, length: int) -> dict[str, np.ndarray]:
    """Triangulated points of the sampled frames inside the GOP."""
    xyz, rgb, tau = [], [], []
    for f in sorted(os.listdir(points_dir)):
        fr = int(f.split("_")[1].split(".")[0])
        if start <= fr < start + length:
            d = np.load(os.path.join(points_dir, f))
            xyz.append(d["xyz"])
            rgb.append(d["rgb"])
            tau.append(np.full(len(d["xyz"]), (fr - start) / length, dtype=np.float32))
    assert xyz, f"no points for GOP [{start}, {start + length}) in {points_dir}"
    return {"xyz": np.concatenate(xyz).astype(np.float32),
            "rgb": np.concatenate(rgb).astype(np.float32) / 255.0,
            "tau": np.concatenate(tau)}
