# Gate 7 (unit, CPU): tracklet confirmation (N_init) in the runtime tracker. PLAN.md §6 Gate 7, decision N6.
import os
import random
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from models.runtime_tracker import RuntimeTracker
from legacy_runtime_tracker import RuntimeTracker as LegacyTracker

K, D, Q = 10, 256, 6


class StubModel(nn.Module):
    """Scripted detector + nearest-box 'ID decoder' oracle. script[t] = list of (cx, cy, w, h, score)."""

    def __init__(self, script):
        super().__init__()
        self.script = script
        self.detr = SimpleNamespace(hidden_dim=D)
        self.id_decoder = None
        self.num_id_vocabulary = K
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, frames=None, part=None, seq_info=None, **kw):
        if part == "detr":
            dets = self.script[frames]
            logits = torch.full((1, Q, 1), -10.0)
            boxes = torch.zeros(1, Q, 4)
            for i, (cx, cy, w, h, s) in enumerate(dets):
                logits[0, i, 0] = torch.logit(torch.tensor(s))
                boxes[0, i] = torch.tensor([cx, cy, w, h])
            return {"pred_logits": logits, "pred_boxes": boxes, "outputs": torch.zeros(1, Q, D),
                    "pos_embeds": torch.zeros(1, Q, D)}
        if part == "trajectory_modeling":
            return seq_info
        if part == "id_decoder":
            tb = seq_info["trajectory_boxes"][0, 0]            # (T, N, 4)
            tm = seq_info["trajectory_masks"][0, 0]
            labels = seq_info["trajectory_id_labels"][0, 0, 0]
            ub = seq_info["unknown_boxes"][0, 0, 0]            # (M, 4)
            M = ub.shape[0]
            probs = torch.full((M, K + 1), 0.01)
            for m in range(M):
                best, best_d = None, 0.15
                for n in range(tb.shape[1]):
                    valid = (~tm[:, n]).nonzero()
                    if len(valid) == 0:
                        continue
                    last = tb[valid[-1, 0], n]
                    d = (last[:2] - ub[m, :2]).abs().max().item()
                    if d < best_d:
                        best, best_d = n, d
                if best is None:
                    probs[m, K] = 0.9
                else:
                    probs[m, int(labels[best])] = 0.9
            return probs.log()[None, None, None], None, None
        raise NotImplementedError(part)


def run(cls, script, **kw):
    tracker = cls(model=StubModel(script), sequence_hw=(100, 100), assignment_protocol="object-max",
                  miss_tolerance=30, det_thresh=0.5, newborn_thresh=0.5, id_thresh=0.1, **kw)
    out = []
    for t in range(len(script)):
        tracker.update(image=t)
        r = tracker.get_track_results()
        out.append({int(i): [round(float(x), 3) for x in b] for i, b in zip(r["id"], r["bbox"])})
    return tracker, out


def obj(t, x0, vx, y=0.5, s=0.9):
    return (x0 + vx * t, y, 0.05, 0.1, s)


def test_consistent_object_confirmed_on_second_frame_with_stable_id():
    script = [[obj(t, 0.2, 0.01)] for t in range(6)]
    _, out1 = run(RuntimeTracker, script, n_init=1)
    _, out2 = run(RuntimeTracker, script, n_init=2)
    assert [len(o) for o in out1] == [1] * 6                  # legacy: emitted immediately
    assert [len(o) for o in out2] == [0] + [1] * 5            # N_init=2: first frame withheld (no back-fill)
    assert len({i for o in out2 for i in o}) == 1             # one stable id for the track
    assert {i for o in out1 for i in o} == {i for o in out2 for i in o}


def test_blip_detection_is_never_emitted_and_slot_is_freed():
    script = [[obj(0, 0.2, 0.0)]] + [[] for _ in range(4)]
    tracker, out = run(RuntimeTracker, script, n_init=2)
    assert all(len(o) == 0 for o in out)
    assert tracker.tentative == {} and tracker.trajectory_id_labels.shape[1] == 0     # context buffer is clean
    assert next(iter(tracker.id_queue)) == 0                                          # freed slot is reused first
    _, out_legacy = run(RuntimeTracker, script, n_init=1)
    assert len(out_legacy[0]) == 1                                                    # N_init=1 would emit the blip


def test_unconfirmed_track_not_reidentified_is_discarded_but_confirmed_survive_occlusion():
    A, B = (0.2, 0.2), (0.6, 0.7)
    script = [
        [(*A, 0.05, 0.1, 0.9), (*B, 0.05, 0.1, 0.9)],   # t0: A, B both newborn (tentative, 1 hit)
        [(*B, 0.05, 0.1, 0.9)],                         # t1: B hit 2 -> confirmed; A missed -> discarded
        [(*A, 0.05, 0.1, 0.9)],                         # t2: A reappears: must be a NEW tentative track; B occluded
        [],                                             # t3: A' missed -> discarded
        [(*B, 0.05, 0.1, 0.9)],                         # t4: B re-identified after 3 missed frames
    ]
    tracker, out = run(RuntimeTracker, script, n_init=2)
    id_B = next(iter(out[1]))
    assert out[0] == {} and len(out[1]) == 1
    assert out[2] == {} and out[3] == {}                # A never confirmed
    assert list(out[4]) == [id_B]                       # confirmed track survives the occlusion with the same id
    emitted = {i for o in out for i in o}
    assert emitted == {id_B}
    assert tracker.tentative == {}


def random_script(seed, T=40, n_obj=5):
    rnd = random.Random(seed)
    objs = [dict(x0=rnd.uniform(0.1, 0.9), y=rnd.uniform(0.1, 0.9), vx=rnd.uniform(-0.01, 0.01),
                 born=rnd.randint(0, 10), die=rnd.randint(20, T)) for _ in range(n_obj)]
    script = []
    for t in range(T):
        dets = []
        for o in objs:
            if o["born"] <= t < o["die"] and not (t % 11 == 7 and rnd.random() < 0.5):
                dets.append((o["x0"] + o["vx"] * t, o["y"], 0.05, 0.1, rnd.uniform(0.6, 0.99)))
        script.append(dets[:Q])
    return script


@pytest.mark.parametrize("seed", range(6))
def test_n_init_1_reproduces_original_motip_tracker_exactly(seed):
    script = random_script(seed)
    _, new = run(RuntimeTracker, script, n_init=1)
    _, old = run(LegacyTracker, script)
    assert new == old


def test_n_init_2_only_withholds_detections_never_adds_new_ones():
    for seed in (1, 3, 5):
        script = random_script(seed)
        _, a = run(RuntimeTracker, script, n_init=1)
        _, b = run(RuntimeTracker, script, n_init=2)
        for t in range(len(script)):                    # every box emitted with N_init=2 is also emitted with N_init=1
            assert sorted(map(tuple, b[t].values())) == sorted(set(map(tuple, b[t].values())))
            assert set(map(tuple, b[t].values())) <= set(map(tuple, a[t].values()))
        assert sum(len(o) for o in b) <= sum(len(o) for o in a)
