"""Export one GOP to the official GIFStream (XDimLab/GIFStream) Neur3D layout, and the test-view GT for re-scoring.

  <out>/png -> <scene>/png, <out>/poses_bounds.npy -> <scene>/poses_bounds.npy   (symlinks)
  <out>/colmap_<gop_start>/sparse/0   SfM of the GOP's first frame with all cameras, one camera id per view
                                      (as GIFStream's own preprocessing; its appearance embedding is per camera id)
  <out>/gt/<prefix>NNNN.png           test views (cam-major, sorted), resized exactly like GIFStream's loader
                                      (cv2 INTER_LINEAR); pass their sorted camera indices as GIFStream's --test_set
"""
from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass

import cv2
import imageio.v2 as imageio
import tyro

from saga.data.prep_points import triangulate_frame


@dataclass
class Args:
    scene_dir: str
    out_dir: str
    gop_start: int = 0
    gop_len: int = 50
    data_factor: int = 2
    test_cams: tuple[str, ...] = ("cam00",)
    gt_prefix: str = "val_step29999_"  # GIFStream names its test renders <stage>_step<step>_<i:04d>.png
    max_image_size: int = 2704


def main(a: Args) -> None:
    os.makedirs(a.out_dir, exist_ok=True)
    for name in ("png", "poses_bounds.npy"):  # poses_bounds only feeds GIFStream's spiral trajectory
        dst = os.path.join(a.out_dir, name)
        if not os.path.lexists(dst) and os.path.exists(os.path.join(a.scene_dir, name)):
            os.symlink(os.path.abspath(os.path.join(a.scene_dir, name)), dst)

    sparse = os.path.join(a.out_dir, f"colmap_{a.gop_start}", "sparse", "0")
    if not os.path.exists(os.path.join(sparse, "points3D.bin")):
        work = tempfile.mkdtemp(prefix=".tri_", dir=a.out_dir)
        try:
            rec = triangulate_frame(a.scene_dir, a.gop_start, (), a.max_image_size, work, per_image_cameras=True)
            os.makedirs(sparse, exist_ok=True)
            rec.write(sparse)
            print(f"sparse: {rec.num_reg_images()} images, {rec.num_cameras()} cameras, {rec.num_points3D()} points")
        finally:
            shutil.rmtree(work, ignore_errors=True)

    gt_dir = os.path.join(a.out_dir, "gt")
    os.makedirs(gt_dir, exist_ok=True)
    test_cams = sorted(os.listdir(os.path.join(a.scene_dir, "png"))) if "all" in a.test_cams else sorted(a.test_cams)
    for k, cam in enumerate(test_cams):
        for i in range(a.gop_len):
            im = imageio.imread(os.path.join(a.scene_dir, "png", cam, f"{a.gop_start + i + 1:05d}.png"))[..., :3]
            im = cv2.resize(im, dsize=(im.shape[1] // a.data_factor, im.shape[0] // a.data_factor),
                            interpolation=cv2.INTER_LINEAR)
            imageio.imwrite(os.path.join(gt_dir, f"{a.gt_prefix}{k * a.gop_len + i:04d}.png"), im)
    print(f"gt: {a.gop_len} images of {test_cams} -> {gt_dir}")


if __name__ == "__main__":
    main(tyro.cli(Args))
