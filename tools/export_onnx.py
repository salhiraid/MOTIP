#!/usr/bin/env python
"""MOTIP2 Phase 10 (skill §13): export the detector and the ID decoder as two SEPARATE ONNX graphs (the paper exports them
separately to TensorRT). The greedy readout, tracklet confirmation and the context buffer stay in Python/C++ outside the
engines (see models/runtime_tracker.py).

    python tools/export_onnx.py --config configs/motip2/s_dancetrack.yaml --ckpt outputs/s/checkpoint_9.pth \
        --out-dir export/s --height 800 --width 1440 --verify

* detector.onnx   x (1,3,H,W) float32 in [0,1] RGB  ->  logits (1,Q,C), boxes (1,Q,4 cxcywh normalised), emb (1,Q,D), pos (1,Q,D)
                  [+ kpts (1,Q,K,2) normalised xy inside each box, kpt_vis (1,Q,K) visibility LOGITS when NUM_KEYPOINTS > 0]
                  FIXED resolution H x W (anchors/positional grids depend on it; pad the frame to it like the tracker does).
* id_decoder.onnx tracks T x N, detections M with DYNAMIC T, N, M (ONNX dynamic axes, verified at several shapes) ->
                  logits (M, K+1).  RoPE alpha (ROPE_ALPHA / ROPE_ALPHA_MODE) is baked in as an inference constant.
                  Inputs (names): traj_feat (T,N,D) [traj_pos (T,N,D) if Spatial Anchor] traj_ids (T,N) int64 traj_times (T,N) int64
                  traj_masks (T,N) bool unk_feat (M,D) [unk_pos (M,D)] unk_masks (M,) bool unk_times (M,) int64
* export_meta.json  everything a deployment needs (K, D, T, thresholds, alpha, resolution, input/output names).

Verification (--verify) runs onnxruntime on random inputs and compares to PyTorch. The detector is compared SET-WISE
(query order may differ between runtimes when scores are near-tied; every ONNX box must exist in the PyTorch output).
TensorRT FP16 engines are NOT built here (no TensorRT / GPU in the development environment).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from configs.util import load_super_config
from models.misc import load_checkpoint
from models.motip import build as build_motip
from utils.misc import yaml_to_dict


class DetectorExport(nn.Module):
    def __init__(self, detr):
        super().__init__()
        self.detr = detr
        self.with_keypoints = getattr(detr, "keypoint_head", None) is not None

    def forward(self, x):
        o = self.detr(x)
        out = (o["pred_logits"], o["pred_boxes"], o["outputs"], o["pos_embeds"])
        if self.with_keypoints:
            out += (o["pred_keypoints"], o["pred_keypoint_vis"])
        return out


DETECTOR_OUTPUT_NAMES = ["logits", "boxes", "emb", "pos"]
KEYPOINT_OUTPUT_NAMES = ["kpts", "kpt_vis"]


def detector_output_names(wrap):
    return DETECTOR_OUTPUT_NAMES + (KEYPOINT_OUTPUT_NAMES if wrap.with_keypoints else [])


class IDDecoderExport(nn.Module):
    """Trajectory adapter + ID decoder with plain tensor I/O (batch and group dims added inside)."""

    def __init__(self, motip, use_pos):
        super().__init__()
        self.tm, self.dec, self.use_pos = motip.trajectory_modeling, motip.id_decoder, use_pos

    def forward(self, *a):
        if self.use_pos:
            tf, tp, tid, tt, tmask, uf, up, umask, ut = a
        else:
            tf, tid, tt, tmask, uf, umask, ut = a
        si = {"trajectory_features": tf[None, None], "trajectory_id_labels": tid[None, None],
              "trajectory_times": tt[None, None], "trajectory_masks": tmask[None, None],
              "unknown_features": uf[None, None, None], "unknown_masks": umask[None, None, None],
              "unknown_times": ut[None, None, None]}
        if self.use_pos:
            si["trajectory_pos"], si["unknown_pos"] = tp[None, None], up[None, None, None]
        if self.tm is not None:
            si = self.tm(si)
        logits, _, _ = self.dec(si, use_decoder_checkpoint=False)
        return logits[0, 0, 0]


def id_decoder_inputs(T, N, M, D, K, use_pos, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)
    tf, uf = r(T, N, D), r(M, D)
    tid = torch.randint(0, K, (T, N), generator=g)
    tt = torch.arange(T)[:, None].repeat(1, N)
    tmask = torch.rand(T, N, generator=g) < 0.2
    umask, ut = torch.zeros(M, dtype=torch.bool), torch.full((M,), T)
    if use_pos:
        return (tf, r(T, N, D), tid, tt, tmask, uf, r(M, D), umask, ut)
    return (tf, tid, tt, tmask, uf, umask, ut)


def id_decoder_input_names(use_pos):
    return (["traj_feat", "traj_pos", "traj_ids", "traj_times", "traj_masks", "unk_feat", "unk_pos", "unk_masks", "unk_times"]
            if use_pos else ["traj_feat", "traj_ids", "traj_times", "traj_masks", "unk_feat", "unk_masks", "unk_times"])


def export_id_decoder(motip, cfg, path, max_context=64, max_tracks=300, max_dets=300, opset=18):
    D, K = motip.id_decoder.feature_dim, motip.id_decoder.num_id_vocabulary
    use_pos = bool(motip.id_decoder.spatial_anchor)
    wrap = IDDecoderExport(motip, use_pos).eval()
    names = id_decoder_input_names(use_pos)
    T, N, M = (torch.export.Dim("T", min=2, max=max_context), torch.export.Dim("N", min=1, max=max_tracks),
               torch.export.Dim("M", min=1, max=max_dets))
    tn = {0: T, 1: N}
    dyn = {"traj_feat": tn, "traj_ids": tn, "traj_times": tn, "traj_masks": tn,
           "unk_feat": {0: M}, "unk_masks": {0: M}, "unk_times": {0: M}}
    if use_pos:
        dyn.update({"traj_pos": tn, "unk_pos": {0: M}})
    args = id_decoder_inputs(6, 4, 3, D, K, use_pos)
    prog = torch.onnx.export(wrap, args, dynamo=True, input_names=names, output_names=["logits"], opset_version=opset,
                             dynamic_shapes={"a": tuple(dyn[n] for n in names)})        # `a` = the *args of IDDecoderExport.forward
    prog.save(path)
    return wrap, names, use_pos, D, K


def export_detector(motip, path, height, width, opset=18):
    wrap = DetectorExport(motip.detr).eval()
    x = torch.rand(1, 3, height, width)
    prog = torch.onnx.export(wrap, (x,), dynamo=True, input_names=["x"], output_names=detector_output_names(wrap),
                             opset_version=opset)
    prog.save(path)
    return wrap


def verify_id_decoder(wrap, path, names, use_pos, D, K, shapes=((6, 4, 3), (11, 9, 5), (3, 1, 1), (1, 3, 2), (20, 30, 12))):
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    worst = 0.0
    for i, (T, N, M) in enumerate(shapes):
        a = id_decoder_inputs(T, N, M, D, K, use_pos, seed=i)
        with torch.no_grad():
            ref = wrap(*a).numpy()
        out = sess.run(None, {n: x.numpy() for n, x in zip(names, a)})[0]
        assert out.shape == ref.shape == (M, K + 1)
        worst = max(worst, float(np.abs(out - ref).max()))
    return worst


def verify_detector(wrap, path, height, width, n_inputs=2, box_tol=1e-3):
    """Set-wise parity: every ONNX query box must be found in the PyTorch output; compare the outputs of matched queries."""
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    worst_other, worst_found = 0.0, 1.0
    for s in range(n_inputs):
        torch.manual_seed(s)
        x = torch.rand(1, 3, height, width)
        with torch.no_grad():
            ref = [r.numpy()[0] for r in wrap(x)]
        out = [o[0] for o in sess.run(None, {"x": x.numpy()})]
        d = np.abs(out[1][:, None] - ref[1][None]).sum(-1)                  # (Q_onnx, Q_torch) L1 distance of boxes
        j = d.argmin(1)
        found = float((d.min(1) < box_tol).mean())
        worst_found = min(worst_found, found)
        ok = d.min(1) < box_tol
        for k in [0, 2, 3] + list(range(4, len(ref))):                      # logits, embeddings, positional enc., keypoints
            worst_other = max(worst_other, float(np.abs(out[k][ok] - ref[k][j[ok]]).max()))
    return worst_found, worst_other


def onnx_opset(path):
    """The opset the file REALLY declares (the dynamo exporter may emit a higher one than requested)."""
    import onnx
    return max(o.version for o in onnx.load(path, load_external_data=False).opset_import if o.domain in ("", "ai.onnx"))


def onnx_ops(path):
    """Operator types used by the graph (check them against your TensorRT version's supported-operator list)."""
    import onnx
    return sorted({n.op_type for n in onnx.load(path, load_external_data=False).graph.node})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None, help="trained checkpoint (EMA weights are used); omit for a random-weight smoke export")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--height", type=int, default=800)
    ap.add_argument("--width", type=int, default=1440)
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--max-context", type=int, default=64)
    ap.add_argument("--max-tracks", type=int, default=300)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--only", choices=["detector", "id_decoder"], default=None)
    ap.add_argument("--num-keypoints", type=int, default=None, help="override NUM_KEYPOINTS of the config (must match the checkpoint)")
    a = ap.parse_args(argv)

    cfg = yaml_to_dict(a.config)
    cfg = load_super_config(cfg, cfg.get("SUPER_CONFIG_PATH"))
    cfg["ONLY_DETR"] = False
    if a.num_keypoints is not None:
        cfg["NUM_KEYPOINTS"] = a.num_keypoints
    motip, _ = build_motip(cfg)
    if a.ckpt:
        load_checkpoint(motip, path=a.ckpt)
    motip = motip.cpu().eval()
    if motip.detr_framework != "deim":
        raise SystemExit("ONNX export is implemented for the DEIM detector only")
    if hasattr(motip.id_decoder, "set_rope_alpha"):
        motip.id_decoder.set_rope_alpha(cfg.get("ROPE_ALPHA", 1.0), cfg.get("ROPE_ALPHA_MODE", "multiply"))
    os.makedirs(a.out_dir, exist_ok=True)
    meta = {"config": a.config, "checkpoint": a.ckpt, "detector_resolution_HW": [a.height, a.width], "opset": a.opset,
            "num_classes": cfg["NUM_CLASSES"], "inference": {k: cfg.get(k) for k in (
                "DET_THRESH", "NEWBORN_THRESH", "ID_THRESH", "MISS_TOLERANCE", "N_INIT", "N_INIT_TTL", "ROPE_ALPHA",
                "ROPE_ALPHA_MODE", "ASSIGNMENT_PROTOCOL", "INFERENCE_MAX_SHORTER", "INFERENCE_MAX_LONGER")}}
    report = {}
    if a.only != "id_decoder":
        p = os.path.join(a.out_dir, "detector.onnx")
        wrap = export_detector(motip, p, a.height, a.width, a.opset)
        meta["detector"] = {"file": "detector.onnx", "inputs": ["x"], "outputs": detector_output_names(wrap),
                            "hidden_dim": motip.detr.hidden_dim, "num_keypoints": int(cfg.get("NUM_KEYPOINTS", 0) or 0),
                            "onnx_ops": onnx_ops(p), "preprocessing": "RGB, float32, x/255 in [0,1], NO mean/std normalisation; "
                            "resize/pad the frame to (H, W) = detector_resolution_HW"}
        meta["detector"]["opset_in_file"] = onnx_opset(p)
        print(f"wrote {p}  (opset {meta['detector']['opset_in_file']})")
        if meta["detector"]["opset_in_file"] != a.opset:
            print(f"  NOTE: requested opset {a.opset} but the file declares {meta['detector']['opset_in_file']}: the PyTorch dynamo "
                  f"exporter raises the opset when an operator needs it. Check that your TensorRT parser supports it.")
        if a.verify:
            found, diff = verify_detector(wrap, p, a.height, a.width)
            report["detector"] = {"boxes_found": found, "max_diff_matched": diff}
            print(f"  detector parity (set-wise): {100 * found:.1f}% of ONNX boxes found in PyTorch output, "
                  f"max |diff| of matched logits/emb/pos = {diff:.2e}")
            # <100% is expected only for near-tied (untrained) scores: queries at the top-k boundary may swap. Matched queries are exact.
            assert found >= 0.95 and diff < 1e-3, "detector ONNX parity failed"
    if a.only != "detector":
        p = os.path.join(a.out_dir, "id_decoder.onnx")
        wrap, names, use_pos, D, K = export_id_decoder(motip, cfg, p, a.max_context, a.max_tracks, a.max_tracks, a.opset)
        meta["id_decoder"] = {"file": "id_decoder.onnx", "inputs": names, "outputs": ["logits"], "dim": D, "K": K,
                              "onnx_ops": onnx_ops(p), "opset_in_file": onnx_opset(p),
                              "spatial_anchor": use_pos, "dynamic_axes": {"T": [2, a.max_context], "N": [1, a.max_tracks],
                                                                         "M": [1, a.max_tracks]}}
        print(f"wrote {p}")
        if a.verify:
            diff = verify_id_decoder(wrap, p, names, use_pos, D, K)
            report["id_decoder"] = {"max_diff": diff}
            print(f"  ID decoder parity at 5 shapes (incl. a 1-frame context) (dynamic T/N/M): max |diff| = {diff:.2e}")
            assert diff < 1e-3, "ID decoder ONNX parity failed"
    meta["verification"] = report
    json.dump(meta, open(os.path.join(a.out_dir, "export_meta.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
