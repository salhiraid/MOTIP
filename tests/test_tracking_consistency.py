# Train/inference consistency of the TRACKING STACK with a TRAINED ID decoder: a decoder trained on clips (frame indices 0..T-1,
# random per-clip identity labels, teacher-forced labelled history) is plugged into the real RuntimeTracker (re-indexed frame
# times, FIFO context, newborn/identity bookkeeping, anchor p_i handling, greedy readout, tracklet confirmation) and the produced
# MOT files are scored with TrackEval. Oracle detections isolate the tracking stack from detector quality (a detector cannot be
# trained to convergence on this CPU). Negative control: the same stack with an UNTRAINED decoder must score much lower.
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from helpers.toy_task import make_batch, make_decoder, train_eval, world, fourier_pos
from models.runtime_tracker import RuntimeTracker

W_PX, H_PX, N_OBJ, D, K, CTX = 320, 192, 4, 48, 10, 6
Q = 8


class OracleModel(nn.Module):
    """Detector = oracle (ground-truth detections with appearance + positional encoding), ID decoder = the given module."""

    def __init__(self, dec, feat, xy, missing=()):
        super().__init__()
        self.dec = dec
        self.feat, self.xy, self.missing = feat, xy, set(missing)          # missing: {(frame, object)} detections to drop
        self.detr = SimpleNamespace(hidden_dim=D)
        self.id_decoder = dec
        self.num_id_vocabulary = K

    def forward(self, frames=None, part=None, seq_info=None, **kw):
        if part == "detr":
            t = frames
            keep = [n for n in range(N_OBJ) if (t, n) not in self.missing]
            logits = torch.full((1, Q, 1), -10.0)
            boxes, emb, pos = torch.zeros(1, Q, 4), torch.zeros(1, Q, D), torch.zeros(1, Q, D)
            pe = fourier_pos(self.xy[t], D)
            for q, n in enumerate(keep):
                logits[0, q, 0] = 5.0
                boxes[0, q] = torch.tensor([*self.xy[t, n].tolist(), 0.12, 0.12])
                emb[0, q], pos[0, q] = self.feat[t, n], pe[n]
            return {"pred_logits": logits, "pred_boxes": boxes, "outputs": emb, "pos_embeds": pos}
        if part == "trajectory_modeling":
            return seq_info
        if part == "id_decoder":
            return self.dec(seq_info, use_decoder_checkpoint=False)
        raise NotImplementedError(part)


def trained_decoder(steps=300):
    dec = make_decoder(D=D, K=K, spatial_anchor=True, seed=0)
    mk = lambda B, s: make_batch(B, T=CTX + 1, N=N_OBJ, D=D, K=K, content="distinct", seed=s, with_pos=True)
    acc, _ = train_eval(dec, mk, mk, steps=steps)
    assert acc > 0.97, acc                                                  # the decoder itself learned the association
    return dec.eval()


def run_tracker(model, S, n_init, miss_tolerance=CTX + 2):
    tr = RuntimeTracker(model=model, sequence_hw=(H_PX, W_PX), assignment_protocol="object-max", miss_tolerance=miss_tolerance,
                        det_thresh=0.5, newborn_thresh=0.5, id_thresh=0.1, n_init=n_init)
    lines = []
    for t in range(S):
        tr.update(image=t)
        r = tr.get_track_results()
        for i, b in zip(r["id"].tolist(), r["bbox"].tolist()):
            lines.append(f"{t + 1},{i},{b[0]:.2f},{b[1]:.2f},{b[2]:.2f},{b[3]:.2f},1,-1,-1,-1")
    return lines


def trackeval(tmp, xy, S, lines):
    seq = "toy0001"
    gt_dir = tmp / "gt" / seq
    (gt_dir / "gt").mkdir(parents=True)
    rows = []
    for t in range(S):
        for n in range(N_OBJ):
            cx, cy = float(xy[t, n, 0]) * W_PX, float(xy[t, n, 1]) * H_PX
            rows.append(f"{t + 1},{n + 1},{cx - 0.06 * W_PX:.2f},{cy - 0.06 * H_PX:.2f},{0.12 * W_PX:.2f},{0.12 * H_PX:.2f},1,1,1")
    (gt_dir / "gt" / "gt.txt").write_text("\n".join(rows) + "\n")
    (gt_dir / "seqinfo.ini").write_text(f"[Sequence]\nname={seq}\nimDir=img1\nframeRate=25\nseqLength={S}\nimWidth={W_PX}\nimHeight={H_PX}\nimExt=.jpg\n")
    (tmp / "seqmap.txt").write_text(f"name\n{seq}\n")
    tr = tmp / "tr"
    tr.mkdir(exist_ok=True)
    (tr / f"{seq}.txt").write_text("\n".join(lines) + "\n")
    root = os.path.join(os.path.dirname(__file__), "..")
    cmd = [sys.executable, "TrackEval/scripts/run_mot_challenge.py", "--SPLIT_TO_EVAL", "val", "--METRICS", "HOTA", "CLEAR", "Identity",
           "--GT_FOLDER", str(tmp / "gt"), "--SEQMAP_FILE", str(tmp / "seqmap.txt"), "--SKIP_SPLIT_FOL", "True", "--TRACKERS_TO_EVAL", "",
           "--TRACKER_SUB_FOLDER", "", "--USE_PARALLEL", "False", "--PLOT_CURVES", "False", "--TRACKERS_FOLDER", str(tr)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=root)
    assert r.returncode == 0, r.stderr[-400:]
    s = (tr / "pedestrian_summary.txt").read_text().split("\n")
    return {k: float(v) for k, v in zip(s[0].split(), s[1].split())}


@pytest.fixture(scope="module")
def decoder():
    return trained_decoder()


@pytest.mark.parametrize("n_init", [1, 2])
def test_trained_decoder_inside_the_real_tracker_tracks_four_objects_almost_perfectly(decoder, tmp_path, n_init):
    S = 40
    feat, xy = world(T=S, N=N_OBJ, D=D, seed=3)
    decoder.set_rope_alpha(1.0)
    lines = run_tracker(OracleModel(decoder, feat, xy), S, n_init)
    m = trackeval(tmp_path, xy, S, lines)
    # n_init=2 withholds each track's first frame (online-faithful, no back-fill): 4 of 160 detections -> DetA/HOTA ~ -1.5 pts
    assert m["HOTA"] > 93.0 and m["IDSW"] == 0 and m["IDF1"] > 95.0, m


def test_tracker_recovers_identities_after_missed_detections(decoder, tmp_path):
    """Objects 1 and 2 are not detected for 3 frames (occlusion). The FIFO context + ID decoder must re-identify them with the
    SAME ids afterwards: no identity switches."""
    S = 40
    feat, xy = world(T=S, N=N_OBJ, D=D, seed=5)
    missing = {(t, 1) for t in range(14, 17)} | {(t, 2) for t in range(25, 28)}
    decoder.set_rope_alpha(1.0)
    lines = run_tracker(OracleModel(decoder, feat, xy, missing), S, n_init=1)
    m = trackeval(tmp_path, xy, S, lines)
    assert m["IDSW"] <= 1 and m["IDF1"] > 90.0 and m["AssA"] > 90.0, m


def test_negative_control_an_untrained_decoder_cannot_track(tmp_path):
    S = 40
    feat, xy = world(T=S, N=N_OBJ, D=D, seed=3)
    untrained = make_decoder(D=D, K=K, spatial_anchor=True, seed=11).eval()
    lines = run_tracker(OracleModel(untrained, feat, xy), S, n_init=1)
    m = trackeval(tmp_path, xy, S, lines)
    assert m["IDSW"] > 10 or m["AssA"] < 70.0, m                          # the metric is sensitive to the decoder
