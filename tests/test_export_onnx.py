# Phase 10: ONNX export of detector and ID decoder (separate graphs) + the exported graphs inside the real tracker loop.
import json
import os
import sys

import numpy as np
import pytest
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

pytest.importorskip("onnxruntime")
pytest.importorskip("onnxscript")
import export_onnx as E
from onnx_runtime_model import OnnxMOTIP
from models.motip import build as build_motip
from models.runtime_tracker import RuntimeTracker
from utils.nested_tensor import NestedTensor

H, W = 160, 256


def small_cfg(**over):
    with open(os.path.join(os.path.dirname(__file__), "..", "configs", "motip2", "n_vehicles.yaml")) as f:
        c = yaml.safe_load(f)
    c.update(DETR_NUM_QUERIES=60, NUM_ID_VOCABULARY=20, MISS_TOLERANCE=10, REL_PE_LENGTH=10, **over)
    return c


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    d = tmp_path_factory.mktemp("onnx")
    cfg = small_cfg()
    torch.manual_seed(0)
    motip, _ = build_motip(cfg)
    for p in motip.parameters():                                    # make the (random) network non-degenerate
        if p.requires_grad and p.dim() > 1:
            torch.nn.init.normal_(p, std=0.05)
    ck = d / "m.pth"
    torch.save({"model": motip.state_dict()}, ck)
    cfgp = d / "cfg.yaml"
    cfgp.write_text(yaml.safe_dump(cfg))
    out = d / "export"
    assert E.main(["--config", str(cfgp), "--ckpt", str(ck), "--out-dir", str(out), "--height", str(H), "--width", str(W), "--verify"]) == 0
    return cfg, ck, out


def test_files_and_metadata(exported):
    _, _, out = exported
    meta = json.load(open(out / "export_meta.json"))
    assert (out / "detector.onnx").exists() and (out / "id_decoder.onnx").exists()          # exported SEPARATELY (§13)
    assert meta["detector_resolution_HW"] == [H, W] and meta["id_decoder"]["spatial_anchor"] is True
    assert meta["verification"]["detector"]["boxes_found"] >= 0.95 and meta["verification"]["detector"]["max_diff_matched"] < 1e-3
    assert meta["verification"]["id_decoder"]["max_diff"] < 1e-3
    assert meta["inference"]["ROPE_ALPHA"] == 2.0 and meta["id_decoder"]["K"] == 20


def test_id_decoder_is_dynamic_in_tracks_and_detections(exported):
    import onnxruntime as ort
    cfg, ck, out = exported
    meta = json.load(open(out / "export_meta.json"))["id_decoder"]
    sess = ort.InferenceSession(str(out / "id_decoder.onnx"), providers=["CPUExecutionProvider"])
    motip, _ = build_motip(cfg)
    from models.misc import load_checkpoint
    load_checkpoint(motip, path=str(ck)); motip.eval()
    motip.id_decoder.set_rope_alpha(cfg["ROPE_ALPHA"], cfg["ROPE_ALPHA_MODE"])
    wrap = E.IDDecoderExport(motip, True).eval()
    for T, N, M in [(1, 1, 1), (1, 7, 3), (5, 2, 9), (10, 40, 25)]:                          # none of these was the export shape
        a = E.id_decoder_inputs(T, N, M, meta["dim"], meta["K"], True, seed=T + N)
        got = sess.run(None, {n: x.numpy() for n, x in zip(meta["inputs"], a)})[0]
        assert got.shape == (M, meta["K"] + 1)
        assert np.abs(got - wrap(*a).detach().numpy()).max() < 1e-3


def _same_partition(a, b):
    """Per frame: the same boxes; across the sequence: identities map one-to-one (ids may be renamed)."""
    fwd, bwd = {}, {}
    for fa, fb in zip(a, b):
        ba = {tuple(round(x, 2) for x in box): i for i, box in fa.items()}
        bb = {tuple(round(x, 2) for x in box): i for i, box in fb.items()}
        if set(ba) != set(bb):
            return False
        for box in ba:
            if fwd.setdefault(ba[box], bb[box]) != bb[box] or bwd.setdefault(bb[box], ba[box]) != ba[box]:
                return False
    return True


def test_onnx_graphs_inside_the_real_tracker_match_pytorch(exported):
    """Run RuntimeTracker twice on the same frames: PyTorch model vs the two ONNX graphs. Boxes, number of tracks per
    frame and the identity partition over time must agree (tracker keeps the buffer / readout / confirmation in Python)."""
    cfg, ck, out = exported
    torch_model, _ = build_motip(cfg)
    from models.misc import load_checkpoint
    load_checkpoint(torch_model, path=str(ck)); torch_model.eval()
    torch.manual_seed(1)
    frames = [torch.rand(1, 3, H, W) for _ in range(8)]
    with torch.no_grad():
        scores = torch_model(frames=NestedTensor(frames[0], torch.zeros(1, H, W, dtype=torch.bool)), part="detr")["pred_logits"].sigmoid().max(-1)[0][0]
    thr = float(scores.sort(descending=True)[0][6])                # ~6 detections per frame

    def run(model, **kw):
        tr = RuntimeTracker(model=model, sequence_hw=(H, W), assignment_protocol="object-max", miss_tolerance=10,
                            det_thresh=thr, newborn_thresh=thr, id_thresh=0.1, n_init=1, rope_alpha=cfg["ROPE_ALPHA"],
                            rope_alpha_mode=cfg["ROPE_ALPHA_MODE"], **kw)
        res = []
        for f in frames:
            tr.update(image=NestedTensor(f, torch.zeros(1, H, W, dtype=torch.bool)))
            r = tr.get_track_results()
            res.append({int(i): [float(x) for x in b] for i, b in zip(r["id"], r["bbox"])})
        return res
    ref = run(torch_model)
    got = run(OnnxMOTIP(str(out)))
    assert sum(len(r) for r in ref) > 20                                   # a non-trivial amount of tracking happened
    assert [len(r) for r in ref] == [len(r) for r in got]
    assert _same_partition(ref, got)
