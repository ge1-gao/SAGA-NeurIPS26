"""Training and evaluation of one GOP or a whole sequence (Neu3D layout).

    python -m saga.train --model stg  --scene-dir <scene> --points-dir data/points/<scene> --out-dir results/stg_warm ...
    python -m saga.train --model saga --scene-dir <scene> --points-dir data/points/<scene> --proposal-ckpt results/stg_warm/model.pt
"""
from __future__ import annotations

import copy
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field

import imageio.v2 as imageio
import numpy as np
import torch
import tyro

from saga.anchors.proposal import ProposalConfig
from saga.anchors.tree import TreeConfig
from saga.codec.codec import CodecConfig
from saga.data.neu3d import Neu3DGOP, load_points
from saga.model.saga import SAGAConfig
from saga.utils.metrics import psnr, render_loss, ssim


@dataclass
class Config:
    scene_dir: str
    points_dir: str
    out_dir: str = "results/debug"
    model: str = "stg"
    gop_start: int = 0
    gop_len: int = 50
    downscale: int = 2
    test_cams: tuple[str, ...] = ("cam00",)  # "all": train and test on every view (MIV CTC)
    near_plane: float = 0.01  # rasterizer near plane of the STG warm-up and SAGA
    max_steps: int = 30_000
    batch_views: int = 2
    eta: float = 0.2
    motion_degree: int = 3  # STG only
    eval_every: int = 10_000
    log_every: int = 500
    seed: int = 0
    refine_start: int = 500  # STG densification
    refine_stop: int = 15_000
    reset_every: int = 3_000
    saga: SAGAConfig = field(default_factory=SAGAConfig)
    tree: TreeConfig = field(default_factory=TreeConfig)
    box_pct: float = 0.1
    proposal_ckpt: str = ""  # model.pt of an STG warm-up on the same GOP; with proposal_gop_len, a "{g}" template
    proposal_gop_len: int = 0  # >0: warm-ups of consecutive GOPs of this length, merged (sequence-level model)
    proposal: ProposalConfig = field(default_factory=ProposalConfig)
    cam_color: bool = True  # SAGA: per-training-camera colour affine during training only (identity at test)
    codec: CodecConfig = field(default_factory=CodecConfig)  # SAGA: quantization + estimated rate (--codec.enable)


def make_trainer(cfg: Config, data: Neu3DGOP):
    """Returns (model, step_fn, codec); step_fn(step, batch) -> (loss, rgb); codec is None unless enabled."""
    dev = data.device
    H, W = data.height, data.width

    if cfg.model == "stg":
        from gsplat.strategy import DefaultStrategy
        from saga.model.stg import STGModel
        pts = load_points(cfg.points_dir, cfg.gop_start, cfg.gop_len)
        print(f"init points: {len(pts['xyz'])}")
        xyz, rgb, tau = (torch.from_numpy(pts[k]).to(dev) for k in ("xyz", "rgb", "tau"))
        model = STGModel(xyz, rgb, tau, motion_degree=cfg.motion_degree)
        model.near_plane = cfg.near_plane
        bs = math.sqrt(cfg.batch_views)
        lrs = STGModel.lrs(data.scene_scale)
        optimizers = {k: torch.optim.Adam([p], lr=lrs[k] * bs, eps=1e-15) for k, p in model.params.items()}
        dec_opt = torch.optim.Adam(model.decoder.parameters(), lr=1e-4)
        strategy = DefaultStrategy(refine_start_iter=cfg.refine_start, refine_stop_iter=cfg.refine_stop,
                                   reset_every=cfg.reset_every, verbose=False)
        strategy.check_sanity(model.params, optimizers)
        state = strategy.initialize_state(scene_scale=data.scene_scale)
        sched = torch.optim.lr_scheduler.ExponentialLR(optimizers["means"], gamma=0.01 ** (1.0 / cfg.max_steps))

        def step_fn(step, batch):
            rgb, _, info = model.render(batch, H, W)
            strategy.step_pre_backward(model.params, optimizers, state, step, info)
            loss = render_loss(rgb, batch["image"], cfg.eta)
            loss.backward()
            for opt in list(optimizers.values()) + [dec_opt]:
                opt.step()
                opt.zero_grad(set_to_none=True)
            sched.step()
            strategy.step_post_backward(model.params, optimizers, state, step, info, packed=False)
            return loss, rgb
        return model, step_fn, None

    if cfg.model == "saga":
        from saga.anchors.proposal import GOPView, load_stg, merge_gops, score, select, to_candidates
        from saga.anchors.tree import build_hierarchy
        from saga.model.saga import SAGAModel
        t0 = time.time()
        L = cfg.proposal_gop_len or data.length
        gops = []  # (start frame, STG, scores, keep) per warm-up GOP
        for g in range(0, data.length, L):
            stg = load_stg(cfg.proposal_ckpt.format(g=cfg.gop_start + g), dev)
            stg.near_plane = cfg.near_plane
            scores = score(stg, data if L == data.length else GOPView(data, g, L), cfg.eta)
            keep = select(stg, scores, cfg.proposal)
            gops.append((g, stg, scores, keep))
            print(f"proposal GOP {g}: {len(keep)} Gaussians -> kept {int(keep.sum())} "
                  f"(S/D thresholds {scores['thresholds'][0]:.3g}/{scores['thresholds'][1]:.3g}) [{time.time() - t0:.1f}s]")
        xyz = np.concatenate([s.params["means"][k].detach().cpu().numpy() for _, s, _, k in gops])
        lo, hi = np.percentile(xyz, cfg.box_pct, 0), np.percentile(xyz, 100 - cfg.box_pct, 0)
        center, scale = (lo + hi) / 2, float((hi - lo).max() / 2 * 1.05)
        cam_pos = data.camtoworlds[data.train_idx, :3, 3]
        parts = [(g, to_candidates(s, sc, k, center, scale, cfg.proposal, cam_pos)) for g, s, sc, k in gops]
        cand = parts[0][1] if len(parts) == 1 else merge_gops(parts, L, data.length)
        del gops, stg, scores, parts
        torch.cuda.empty_cache()
        hier = build_hierarchy(cand, cfg.tree, seed=cfg.seed)
        print(f"candidates={len(cand['x'])} (static {cand['static'].mean():.2f}) coarse={len(hier['coarse_center'])} "
              f"fine={len(hier['fine_parent'])} gaussians={len(hier['fine_parent']) * cfg.saga.k_gauss} "
              f"[{time.time() - t0:.1f}s]")
        np.savez(os.path.join(cfg.out_dir, "hierarchy.npz"), box_center=center, box_scale=scale, **hier)
        model = SAGAModel(hier, hier["coarse_h"], center, scale, cfg.saga, dev)
        groups = model.param_groups()
        if cfg.cam_color:
            cam_aff = torch.zeros(len(data.cams), 3, 4, device=dev, requires_grad=True)
            train_idx = torch.tensor(data.train_idx, device=dev)
            groups.append({"name": "cam_color", "params": [cam_aff], "lr": 1e-3, "lr_final": 1e-4})
        codec = None
        if cfg.codec.enable:
            from torch.func import functional_call
            from saga.codec.codec import Codec
            codec = Codec(model, cfg.codec, cfg.tree.bits_f)
            groups.append(codec.param_group())
        opt = torch.optim.Adam([{"params": g["params"], "lr": g["lr"]} for g in groups], eps=1e-15)
        lambdas = [(lambda s, a=g["lr"], b=g["lr_final"]: (b / a) ** min(s / cfg.max_steps, 1.0)) for g in groups]
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambdas)

        def step_fn(step, batch):
            coding = codec is not None and step >= cfg.codec.start
            if coding and step == cfg.codec.start:
                codec.init_steps()
            if step > 0 and step % cfg.saga.knn_refresh == 0 or coding and step == cfg.codec.start:
                if coding:  # the decoder derives the support base from the decoded fine positions
                    model.update_spacing()
                model.refresh_knn()
            if coding:  # render the quantized (relaxed) scaffold, hash and decoders
                codec.schedule(step, cfg.max_steps)
                q = codec.quantize()
                rgb = functional_call(model, q, (batch, H, W))[0]
            else:
                rgb, _, _ = model.render(batch, H, W)
            if cfg.cam_color:  # rgb + rgb·M^T + b; mean over training cameras fixed to identity
                A = (cam_aff - cam_aff[train_idx].mean(0))[batch["cam"]]
                rgb = rgb + torch.einsum("chwj,cij->chwi", rgb, A[..., :3]) + A[:, None, None, :, 3]
            loss = render_loss(rgb, batch["image"], cfg.eta) + model.regularization()
            if coding:
                bits = codec.bits(q)
                codec.last_bits = {k: v.detach() for k, v in bits.items()}
                ramp = min(1.0, (step - cfg.codec.start + 1) / cfg.codec.rate_warmup) if cfg.codec.rate_warmup else 1.0
                loss = loss + ramp * cfg.codec.lambda_rate * sum(bits.values()) / (8 * 2 ** 20)
            loss.backward()
            bad = [n for n, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
            if bad or not torch.isfinite(loss):
                nan_report(model, step, loss, bad)
            opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()
            return loss, rgb
        step_fn.opt, step_fn.sched = opt, sched  # for the divergence rollback in main()
        return model, step_fn, codec
    raise ValueError(cfg.model)


@torch.no_grad()
def nan_report(model, step, loss, bad):
    """Print the state behind the first non-finite loss/gradient, then stop."""
    print(f"\n!!! non-finite at step {step}: loss={loss.item()} bad grads={bad}", flush=True)
    for k, v in {**model.decode(), "res": model.res}.items():
        if torch.is_tensor(v) and v.dtype.is_floating_point:
            fin = torch.isfinite(v)
            a = v[fin].abs()
            rng = f"|v| min={a.min().item():.3g} max={a.max().item():.3g}" if a.numel() else "all non-finite"
            print(f"  {k:10s} non-finite={int((~fin).sum())} {rng}", flush=True)
    raise RuntimeError(f"non-finite loss/gradient at step {step}")


def n_gaussians(model) -> int:
    return len(model.params["means"]) if hasattr(model, "params") else model.num_gaussians()


@torch.no_grad()
def evaluate(cfg: Config, model, data: Neu3DGOP, tag: str) -> dict:
    import lpips
    lp = lpips.LPIPS(net="vgg", verbose=False).to(data.device)
    H, W = data.height, data.width
    out_dir = os.path.join(cfg.out_dir, f"renders_{tag}")
    os.makedirs(out_dir, exist_ok=True)
    ps, ss, ls = [], [], []
    idx = torch.tensor(data.test_idx, device=data.device)
    torch.cuda.synchronize()
    t0 = time.time()
    for t in range(data.length):
        b = data.batch(idx, t)
        rgb = model.render(b, H, W)[0].clamp(0, 1)
        gt = b["image"]
        p, g = rgb.permute(0, 3, 1, 2), gt.permute(0, 3, 1, 2)
        ps.append(psnr(rgb, gt).item())
        ss.append(ssim(p, g).item())
        ls.append(lp(p * 2 - 1, g * 2 - 1).mean().item())
        if t % 10 == 0:
            imageio.imwrite(os.path.join(out_dir, f"{t:03d}.png"),
                            (torch.cat([rgb[0], gt[0]], 1).cpu().numpy() * 255).astype(np.uint8))
    torch.cuda.synchronize()
    eval_time = time.time() - t0
    b = data.batch(idx[:1], 0)
    torch.cuda.synchronize()
    t0 = time.time()
    for t in range(data.length):
        b["tau"] = data.time_of(t)
        model.render(b, H, W)
    torch.cuda.synchronize()
    fps = data.length / (time.time() - t0)
    res = {"psnr": float(np.mean(ps)), "ssim": float(np.mean(ss)), "lpips_vgg": float(np.mean(ls)),
           "fps": fps, "eval_time_s": eval_time, "stored_MB_fp32": sum(p.numel() * 4 for p in model.parameters()) / 2**20,
           "psnr_per_frame": ps, "num_gaussians": n_gaussians(model)}
    if hasattr(model, "stored_floats"):
        floats = model.stored_floats()
        res["stored_MB_fp32_by_component"] = {k: v * 4 / 2**20 for k, v in floats.items()}
        res["stored_MB_fp32"] = sum(floats.values()) * 4 / 2**20
    return res


def main(cfg: Config):
    torch.manual_seed(cfg.seed)
    cfg.saga.near_plane = cfg.near_plane
    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    t_load = time.time()
    data = Neu3DGOP(cfg.scene_dir, cfg.gop_start, cfg.gop_len, cfg.downscale, cfg.test_cams)
    print(f"loaded {data.images.shape} in {time.time() - t_load:.1f}s; scene_scale={data.scene_scale:.3f}")
    model, step_fn, codec = make_trainer(cfg, data)

    # Divergence rollback (SAGA, before the codec starts): the model and optimizer state is snapshotted every 250
    # healthy steps; if the median train PSNR of the last 50 steps drops 5 dB below the snapshot's (or below 10 dB
    # after step 300), training restarts from the snapshot with a different batch order.
    guard = hasattr(step_fn, "opt")
    guard_end = cfg.codec.start if codec is not None else cfg.max_steps
    hist, snap, rollbacks = [], None, 0

    def take_snapshot(step):
        return {"step": step, "psnr": float(np.median(hist[-100:])) if hist else 0.0,
                "model": {k: v.detach().clone() for k, v in model.state_dict().items()},
                "opt": copy.deepcopy(step_fn.opt.state_dict()), "sched": step_fn.sched.state_dict()}

    t0 = time.time()
    step = 0
    while step < cfg.max_steps:
        batch = data.sample(cfg.batch_views)
        loss, rgb = step_fn(step, batch)
        if guard and step < guard_end:
            hist.append(psnr(rgb.detach(), batch["image"]).item())
            if step % 250 == 0 and (snap is None or np.median(hist[-100:]) > snap["psnr"] - 1):
                snap = take_snapshot(step)
            elif len(hist) >= 50 and step - snap["step"] >= 50 and (
                    np.median(hist[-50:]) < snap["psnr"] - 5 or step >= 300 and np.median(hist[-50:]) < 10):
                rollbacks += 1
                if rollbacks > 10:
                    raise RuntimeError(f"diverged {rollbacks} times; last good step {snap['step']}")
                print(f"!!! divergence at step {step}: train PSNR median {np.median(hist[-50:]):.2f} vs "
                      f"{snap['psnr']:.2f} at step {snap['step']}; rollback #{rollbacks}", flush=True)
                model.load_state_dict(snap["model"])
                step_fn.opt.load_state_dict(copy.deepcopy(snap["opt"]))
                step_fn.sched.load_state_dict(snap["sched"])
                model.refresh_knn()
                torch.manual_seed(cfg.seed + 1000 * rollbacks)
                step, hist = snap["step"] + 1, hist[:0]
                continue
        if step % cfg.log_every == 0:
            rate = ""
            if codec is not None and codec.last_bits is not None:
                mb = {k: v.item() / 8 / 2**20 for k, v in codec.last_bits.items()}
                rate = f" rate={sum(mb.values()):.3f}MB (" + " ".join(f"{k}={v:.3f}" for k, v in mb.items()) + ")"
            print(f"[{step:6d}] loss={loss.item():.4f} psnr={psnr(rgb.detach(), batch['image']).item():.2f} "
                  f"N={n_gaussians(model)} t={time.time() - t0:.0f}s{rate}", flush=True)
        if (step + 1) % cfg.eval_every == 0 and step + 1 < cfg.max_steps:
            r = evaluate(cfg, model, data, f"{step + 1}")
            rate = ""
            if codec is not None and step >= cfg.codec.start:  # hard-quantized rate of the current state
                with torch.no_grad():
                    mb = {k: v.item() / 8 / 2**20 for k, v in codec.bits(codec.quantize(False), False).items()}
                rate = f" rate={sum(mb.values()):.3f}MB (" + " ".join(f"{k}={v:.3f}" for k, v in mb.items()) + ")"
            print(f"[eval {step + 1}] psnr={r['psnr']:.3f} ssim={r['ssim']:.4f} lpips={r['lpips_vgg']:.4f}{rate}",
                  flush=True)
        step += 1
    if rollbacks:
        print(f"divergence rollbacks: {rollbacks}", flush=True)
    train_time = time.time() - t0

    rate = codec.finalize() if codec is not None else None  # the model now holds the decoded values
    r = evaluate(cfg, model, data, "final")
    r["train_time_s"] = train_time
    msg = ""
    if rate is not None:
        r["rate_MB"] = sum(rate.values())
        r["rate_MB_by_component"] = rate
        msg = f" rate={r['rate_MB']:.3f}MB"
        torch.save(codec.state_dict(), os.path.join(cfg.out_dir, "codec.pt"))
    print(f"[final] psnr={r['psnr']:.3f} ssim={r['ssim']:.4f} lpips={r['lpips_vgg']:.4f} fps={r['fps']:.1f} "
          f"size={r['stored_MB_fp32']:.1f}MB{msg} N={r['num_gaussians']} train={train_time / 60:.1f}min")
    if rate is not None:
        print("  rate by component (MB): " + " ".join(f"{k}={v:.4f}" for k, v in rate.items()))
    with open(os.path.join(cfg.out_dir, "metrics.json"), "w") as f:
        json.dump(r, f, indent=2)
    torch.save(model.state_dict(), os.path.join(cfg.out_dir, "model.pt"))


if __name__ == "__main__":
    main(tyro.cli(Config))
