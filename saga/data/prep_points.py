"""Per-frame SfM triangulation with the fixed Neu3D poses, test cameras excluded -> <out_dir>/frame_XXXXX.npz."""
from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass, field

import numpy as np
import pycolmap
import tyro

from saga.data.neu3d import read_cameras


@dataclass
class Args:
    scene_dir: str
    out_dir: str
    frames: list[int] = field(default_factory=lambda: list(range(0, 300, 5)))
    test_cams: tuple[str, ...] = ("cam00",)
    max_image_size: int = 2704
    min_track_len: int = 3


def triangulate_frame(scene_dir: str, frame: int, test_cams, max_image_size: int, work: str,
                      per_image_cameras: bool = False) -> pycolmap.Reconstruction:
    cams = read_cameras(os.path.join(scene_dir, "colmap_0", "manual"))
    names = sorted(n for n in cams if n not in test_cams and os.path.isdir(os.path.join(scene_dir, "png", n)))
    img_dir = os.path.join(work, "images")
    os.makedirs(img_dir)
    for n in names:
        os.symlink(os.path.join(scene_dir, "png", n, f"{frame + 1:05d}.png"), os.path.join(img_dir, f"{n}.png"))
    db_path = os.path.join(work, "db.db")

    c0 = cams[names[0]]
    per_image_cameras = per_image_cameras or any(not np.allclose(cams[n].K, c0.K) for n in names)
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "PINHOLE"
    reader.camera_params = ",".join(str(v) for v in (c0.K[0, 0], c0.K[1, 1], c0.K[0, 2], c0.K[1, 2]))
    sift = pycolmap.SiftExtractionOptions()
    sift.max_image_size = max_image_size
    pycolmap.extract_features(db_path, img_dir, image_names=[f"{n}.png" for n in names],
                              camera_mode=pycolmap.CameraMode.PER_IMAGE if per_image_cameras else pycolmap.CameraMode.SINGLE,
                              camera_model="PINHOLE", reader_options=reader,
                              sift_options=sift)
    if per_image_cameras:  # the reader seeded every camera with the first view's K; set each view's own before matching
        db = pycolmap.Database()
        db.open(db_path)
        cam_name = {im.camera_id: os.path.splitext(im.name)[0] for im in db.read_all_images()}
        for cam in db.read_all_cameras():
            K = cams[cam_name[cam.camera_id]].K
            cam.params = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])
            db.update_camera(cam)
        db.close()
    pycolmap.match_exhaustive(db_path)

    rec = pycolmap.Reconstruction()
    db = pycolmap.Database()
    db.open(db_path)
    images = db.read_all_images()
    name_of_image = {im.image_id: im.name for im in images}
    frame_of_image = {}
    for cam in db.read_all_cameras():
        rec.add_camera(cam)
    for rig in db.read_all_rigs():
        rec.add_rig(rig)
    for fr in db.read_all_frames():
        (data_id,) = list(fr.data_ids)
        frame_of_image[data_id.id] = fr.frame_id
        vm = cams[os.path.splitext(name_of_image[data_id.id])[0]].viewmat
        fr.rig_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(vm[:3, :3]), vm[:3, 3])
        rec.add_frame(fr)
    for im in images:
        im.frame_id = frame_of_image[im.image_id]
        rec.add_image(im)
    db.close()
    out = os.path.join(work, "tri")
    os.makedirs(out)
    return pycolmap.triangulate_points(rec, db_path, img_dir, out)


def main(a: Args):
    os.makedirs(a.out_dir, exist_ok=True)
    for fr in a.frames:
        dst = os.path.join(a.out_dir, f"frame_{fr:05d}.npz")
        if os.path.exists(dst):
            continue
        work = tempfile.mkdtemp(prefix=f".tri_{fr}_", dir=a.out_dir)
        try:
            rec = triangulate_frame(a.scene_dir, fr, a.test_cams, a.max_image_size, work)
            pts = [p for p in rec.points3D.values() if p.track.length() >= a.min_track_len]
            xyz = np.array([p.xyz for p in pts], dtype=np.float32)
            rgb = np.array([p.color for p in pts], dtype=np.uint8)
            np.savez(dst, xyz=xyz, rgb=rgb)
            print(f"frame {fr}: {len(xyz)} points", flush=True)
        finally:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
