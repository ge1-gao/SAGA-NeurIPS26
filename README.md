<div align="center">

# SAGA: Gaussian Splatting-based Volumetric Video Compression with Sparse 4D Anchors

**NeurIPS 2026**

[![arXiv](https://img.shields.io/badge/arXiv-2609.33969-b31b1b.svg)](https://arxiv.org/abs/2609.33969)
[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS-2026-4b44ce.svg)](https://neurips.cc/)

</div>

This repository contains implementation of the **representation** (training and rendering) and **compression** modules of SAGA. 

## to-dos

- [ ] descriptions for Panoptic Sports and MPEG CTC.
- [ ] finalize the compression section of the code.

## installation

```bash
git clone https://github.com/ge1-gao/SAGA-NeurIPS26.git saga && cd saga
conda create -n saga python=3.9 -y && conda activate saga
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
pip install gsplat==1.5.3 tyro imageio pillow lpips plyfile opencv-python pytest "numpy<2"
conda install -c conda-forge pycolmap   # for SfM point triangulation
```
gsplat compiles its CUDA kernels on first use, so you need a GPU machine with the CUDA toolkit (tested with CUDA 12.6).

## data preparation

Download [Neural 3D Video](https://github.com/facebookresearch/Neural_3D_Video). Each scene folder contains `cam00.mp4 … cam20.mp4` and `poses_bounds.npy`.

```bash
SCENE=/path/to/Neural_3D_Video/flame_steak

# 1. extract frames and cameras
python -m saga.data.prep_neu3d --scene-dir $SCENE --n-frames 50

# 2. triangulate SfM points on every 5th frame (test view cam00 excluded)
python -m saga.data.prep_points --scene-dir $SCENE --out-dir data/points/flame_steak \
    --frames 0 5 10 15 20 25 30 35 40 45
```

Expected layout:

```
$SCENE/
├── png/cam00/00001.png ...       # frames (1-indexed)
└── colmap_0/manual/              # cameras.txt, images.txt
data/points/flame_steak/
└── frame_00000.npz ...           # SfM points per sampled frame
```

## optimization

**Step 1 — STG warm-up** (~4k steps). Used only to propose anchor candidates:

```bash
python -m saga.train --model stg --scene-dir $SCENE --points-dir data/points/flame_steak \
    --out-dir results/stg_warm --max-steps 4000 --refine-stop 3500 --reset-every 100000 --eval-every 100000
```

**Step 2 — SAGA** (30k steps):

```bash
python -m saga.train --model saga --scene-dir $SCENE --points-dir data/points/flame_steak \
    --proposal-ckpt results/stg_warm/model.pt --out-dir results/saga
```

Colour compensation (on by default, `--no-cam-color` to disable) learns a 3×4 colour affine per training camera that absorbs the colour inconsistency between Neu3D cameras. The mean over training cameras is fixed to the identity. It is used only during training: test views are rendered without it, and it is not stored in `model.pt` or counted in the model size. On flame_steak GOP0 it raises cam00 PSNR from 33.21 to 34.46 dB.

Evaluation on `cam00` runs automatically at the end. `--out-dir` then contains:

| File | Content |
|---|---|
| `metrics.json` | PSNR, SSIM, LPIPS-VGG, per-frame PSNR, FPS, model size by component |
| `renders_final/` | test renders (left: ours, right: GT) |
| `hierarchy.npz` | the anchor hierarchy |
| `model.pt` | trained model |

Common options (see `python -m saga.train --help` for all of them):

| Option | Meaning | Default |
|---|---|---|
| `--gop-start` | first frame of the GOP | `0` |
| `--tree.n-coarse` / `--tree.n-fine` | number of coarse / fine anchors | `4000` / `30000` |
| `--saga.k-gauss` | Gaussians per fine anchor | `10` |
| `--no-cam-color` | disable the per-training-camera colour compensation | on |
| `--saga.use-hash` | enable the residual hash (`--saga.hash-type` = `local` or `grid`, `--saga.hash-table` = size) | off |

To re-score renders from another method with our metrics:

```bash
python -m saga.eval_renders --render-dir <renders> --gt-dir <gt>
```

To run the tests:

```bash
python -m pytest -q tests
```

## Code structure

```
saga/
├── train.py              # per-GOP training / evaluation (stg, saga)
├── eval_renders.py       # re-score external renders
├── model/
│   ├── saga.py           # SAGA model: hierarchy decoding + rendering
│   ├── stg.py            # STG (warm-up)
│   ├── temporal.py       # temporal opacity, polynomial motion
│   └── embed.py          # positional encoding, residual hash (local / grid)
├── anchors/
│   ├── proposal.py       # warm-up scoring → anchor candidates
│   ├── tree.py           # greedy 4D tree → coarse / fine anchors
│   ├── neighbors.py      # kNN + kernel interpolation
│   └── morton.py         # 4D Morton keys
├── data/
│   ├── neu3d.py          # GOP loader
│   ├── prep_neu3d.py     # frame / camera extraction
│   ├── prep_points.py    # SfM point triangulation
│   ├── export_scaffold4d.py  # data export for 4D Scaffold GS
│   └── export_gifstream.py   # data export for GIFStream
└── utils/metrics.py      # PSNR, SSIM, training loss
tests/                    # unit tests
```

## Acknowledgements

The STG components are ported from [GSCodec Studio](https://github.com/JasonLSC/GSCodec_Studio). Rendering uses [gsplat](https://github.com/nerfstudio-project/gsplat). 

## Citation

```bibtex
@inproceedings{saga2026,
  title     = {Gaussian Splatting-based Volumetric Video Compression with Sparse 4D Anchors},
  author    = {Gao, Ge and Teng, Siyue and Wang, Chanqgi and Zhang, Fan and Anantrasirichai, Nantheera and Chiang, Jui Chiu and Peng, Wen-Hsiao and Bull, David},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```
