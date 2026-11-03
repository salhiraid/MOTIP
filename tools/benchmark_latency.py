# MOTIP2 PLAN Gate 1/2/9: per-component latency in the Tab. 4 protocol (single image 800x1440, ID decoder with T=30
# context frames, 50 trajectories, 50 detections, FP32). Report the ratio MOTIP2-S / MOTIP when not on a V100 (§12).
#   python tools/benchmark_latency.py --config configs/motip2/s_dancetrack.yaml [--ckpt X.pth]
#       [--baseline-config configs/r50_deformable_detr_motip_dancetrack.yaml]
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from configs.util import load_super_config
from models.motip import build as build_motip
from models.misc import load_checkpoint
from utils.misc import yaml_to_dict
from utils.nested_tensor import NestedTensor


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def _time(fn, dev, warmup, iters):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        _sync(dev); t0 = time.perf_counter(); fn(); _sync(dev); ts.append((time.perf_counter() - t0) * 1000)
    t = torch.tensor(ts)
    return t.mean().item(), t.std().item() if len(ts) > 1 else 0.0


def synthetic_seq_info(cfg, dev, ctx, n_tracks, n_dets, dim):
    z = lambda *s: torch.randn(*s, device=dev)
    T, N, M = ctx, n_tracks, n_dets
    si = {
        "trajectory_features": z(1, 1, T, N, dim), "trajectory_boxes": z(1, 1, T, N, 4),
        "trajectory_id_labels": torch.randint(0, cfg["NUM_ID_VOCABULARY"], (1, 1, T, N), device=dev),
        "trajectory_times": torch.arange(T, device=dev)[None, None, :, None].repeat(1, 1, 1, N),
        "trajectory_masks": torch.zeros(1, 1, T, N, dtype=torch.bool, device=dev),
        "unknown_features": z(1, 1, 1, M, dim), "unknown_boxes": z(1, 1, 1, M, 4),
        "unknown_masks": torch.zeros(1, 1, 1, M, dtype=torch.bool, device=dev),
        "unknown_times": torch.full((1, 1, 1, M), T, device=dev),
    }
    if cfg.get("SPATIAL_ANCHOR", False):
        si["trajectory_pos"], si["unknown_pos"] = z(1, 1, T, N, dim), z(1, 1, 1, M, dim)
    return si


@torch.no_grad()
def bench(config_path, ckpt, h, w, ctx, n_tracks, n_dets, warmup, iters):
    cfg = yaml_to_dict(config_path)
    cfg = load_super_config(cfg, cfg.get("SUPER_CONFIG_PATH"))
    cfg["DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(cfg["DEVICE"])
    model, _ = build_motip(cfg)
    if ckpt:
        load_checkpoint(model, path=ckpt)
    model = model.to(dev).eval()
    dim = getattr(model.detr, "hidden_dim", 256)
    x = NestedTensor(torch.rand(1, 3, h, w, device=dev), torch.zeros(1, h, w, dtype=torch.bool, device=dev))
    si = synthetic_seq_info(cfg, dev, ctx, n_tracks, n_dets, dim)
    det = _time(lambda: model(frames=x, part="detr"), dev, warmup, iters)

    def _id():
        s = model(seq_info=dict(si), part="trajectory_modeling")
        return model(seq_info=s, part="id_decoder")
    idd = _time(_id, dev, warmup, iters)
    return {"device": str(dev), "detector_ms": det, "id_decoder_ms": idd, "total_ms": det[0] + idd[0],
            "params_M": sum(p.numel() for p in model.parameters()) / 1e6}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--baseline-config", default=None)
    ap.add_argument("--height", type=int, default=800)
    ap.add_argument("--width", type=int, default=1440)
    ap.add_argument("--context", type=int, default=30)
    ap.add_argument("--num-tracks", type=int, default=50)
    ap.add_argument("--num-dets", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    a = ap.parse_args()
    args = (a.height, a.width, a.context, a.num_tracks, a.num_dets, a.warmup, a.iters)
    r = bench(a.config, a.ckpt, *args)
    print(f"[{a.config}] device={r['device']} params={r['params_M']:.1f}M  detector {r['detector_ms'][0]:.1f}±{r['detector_ms'][1]:.1f} ms"
          f"  ID decoder {r['id_decoder_ms'][0]:.1f}±{r['id_decoder_ms'][1]:.1f} ms  total {r['total_ms']:.1f} ms")
    if a.baseline_config:
        b = bench(a.baseline_config, None, *args)
        print(f"[{a.baseline_config}] total {b['total_ms']:.1f} ms  -> speed-up {b['total_ms'] / r['total_ms']:.2f}x "
              f"(paper: 3.2x for MOTIP2-S vs MOTIP on a V100)")
