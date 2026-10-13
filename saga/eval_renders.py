"""Re-score saved renders with our PSNR / SSIM / LPIPS-VGG ([-1, 1] inputs)."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import torch
import tyro
from PIL import Image

from saga.utils.metrics import psnr, ssim


@dataclass
class Args:
    render_dir: str
    gt_dir: str
    out_json: str = ""  # default: <render_dir>/../metrics_saga.json


def load(path: str, dev: str) -> torch.Tensor:
    return torch.from_numpy(np.asarray(Image.open(path).convert("RGB"))).to(dev).float().div(255)[None]


@torch.no_grad()
def main(a: Args) -> None:
    import lpips
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lp = lpips.LPIPS(net="vgg", verbose=False).to(dev)
    names = sorted(f for f in os.listdir(a.render_dir) if f.endswith(".png"))
    assert names and all(os.path.exists(os.path.join(a.gt_dir, n)) for n in names), "renders and gt must pair by name"
    ps, ss, ls = [], [], []
    for n in names:
        r, g = load(os.path.join(a.render_dir, n), dev), load(os.path.join(a.gt_dir, n), dev)
        rp, gp = r.permute(0, 3, 1, 2), g.permute(0, 3, 1, 2)
        ps.append(psnr(r, g).item())
        ss.append(ssim(rp, gp).item())
        ls.append(lp(rp * 2 - 1, gp * 2 - 1).mean().item())
    res = {"psnr": float(np.mean(ps)), "ssim": float(np.mean(ss)), "lpips_vgg": float(np.mean(ls)), "n_images": len(names)}
    out = a.out_json or os.path.join(os.path.dirname(a.render_dir.rstrip("/")), "metrics_saga.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[rescored] psnr={res['psnr']:.3f} ssim={res['ssim']:.4f} lpips={res['lpips_vgg']:.4f} n={len(names)} -> {out}")


if __name__ == "__main__":
    main(tyro.cli(Args))
