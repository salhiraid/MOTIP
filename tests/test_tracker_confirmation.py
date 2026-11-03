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


def test_detector_only_mode_ignores_n_init():
    """In detector-only evaluation (Gate 2) every detection is newborn each frame; N_init must not suppress them."""
    script = [[obj(t, 0.2, 0.01)] for t in range(5)]
    _, out = run(RuntimeTracker, script, n_init=2, only_detr=True)
    assert [len(o) for o in out] == [1] * 5


def test_unconfirmed_track_ttl_keeps_the_identity_across_a_gap_but_still_requires_consecutive_hits():
    A = (0.3, 0.5, 0.05, 0.1, 0.9)
    script = [[A], [], [A], [A], [A]]            # hit, miss, hit, hit, hit
    t0, out0 = run(RuntimeTracker, script, n_init=2, n_init_ttl=0)
    t2, out2 = run(RuntimeTracker, script, n_init=2, n_init_ttl=2)
    # ttl=0: the track from frame 0 dies at the miss; the re-appearance is a NEW track, confirmed on its 2nd hit (frame 3)
    assert [len(o) for o in out0] == [0, 0, 0, 1, 1]
    # ttl=2: the tentative track survives the gap in the context (same slot is re-identified), but the miss reset the
    # consecutive-hit counter: hit(2) -> hits=1, hit(3) -> hits=2 -> confirmed at frame 3 with the ORIGINAL identity
    assert [len(o) for o in out2] == [0, 0, 0, 1, 1]
    ids0 = {i for o in out0 for i in o}; ids2 = {i for o in out2 for i in o}
    assert len(ids0) == 1 and len(ids2) == 1
    assert next(iter(ids2)) == 0 and next(iter(ids0)) == 1          # ttl=2 kept id 0; ttl=0 had to start a new id (1)
    # a gap longer than the ttl discards the track
    script = [[A], [], [], [], [A], [A]]
    _, out = run(RuntimeTracker, script, n_init=2, n_init_ttl=2)
    assert [len(o) for o in out] == [0, 0, 0, 0, 0, 1] and next(iter(out[-1])) != 0


def _bare_tracker(label_row, id_thresh=0.1, protocol="object-max"):
    tr = RuntimeTracker(model=StubModel([[]]), sequence_hw=(100, 100), assignment_protocol=protocol,
                        miss_tolerance=30, det_thresh=0.5, newborn_thresh=0.5, id_thresh=id_thresh)
    tr.trajectory_id_labels = torch.tensor([label_row])          # identities currently in the context (row 0)
    return tr


@pytest.mark.parametrize("protocol", ["object-max", "id-max", "hungarian"])
def test_greedy_readout_gives_unique_known_confident_identities(protocol):
    """Paper §3.1 'greedy protocol converts predicted distributions into unique identity assignments' with a minimum
    identity probability lambda_ID (0.1): no identity is used twice, unknown identities are never assigned, and every
    assigned identity has probability >= lambda_ID; everything else is the newborn class K."""
    g = torch.Generator().manual_seed(0)
    known = [0, 3, 4, 7]
    tr = _bare_tracker(known, id_thresh=0.1, protocol=protocol)
    for _ in range(300):
        M = int(torch.randint(1, 9, (1,), generator=g))
        scores = torch.softmax(3 * torch.randn(M, K + 1, generator=g), dim=-1)
        labels = {"object-max": tr._object_max_assignment, "id-max": tr._id_max_assignment,
                  "hungarian": tr._hungarian_assignment}[protocol](scores)
        assert len(labels) == M
        used = [l for l in labels if l != K]
        assert len(used) == len(set(used)), (protocol, labels)                         # unique
        assert set(used) <= set(known)                                                 # only identities that exist
        for i, l in enumerate(labels):
            if l != K:
                assert scores[i, l] >= 0.1 - 1e-9                                      # lambda_ID


def test_object_max_gives_the_identity_to_the_most_confident_object():
    tr = _bare_tracker([2, 5], id_thresh=0.1)
    s = torch.full((3, K + 1), 0.001)
    s[0, 2], s[1, 2], s[2, 5] = 0.9, 0.6, 0.8            # objects 0 and 1 both want id 2; 0 is more confident
    s = s / s.sum(-1, keepdim=True)
    assert tr._object_max_assignment(s) == [2, K, 5]


def test_fifo_context_never_exceeds_the_miss_tolerance():
    T = 10
    script = [[obj(t, 0.1, 0.005), obj(t, 0.6, -0.005, y=0.2)] for t in range(60)]
    tracker = RuntimeTracker(model=StubModel(script), sequence_hw=(100, 100), assignment_protocol="object-max",
                             miss_tolerance=T, det_thresh=0.5, newborn_thresh=0.5, id_thresh=0.1, n_init=2)
    sizes = []
    for t in range(len(script)):
        tracker.update(image=t)
        sizes.append(tracker.trajectory_features.shape[0])
    assert max(sizes) <= T and sizes[-1] >= T - 2                  # a bounded FIFO that is actually full in steady state
    ids = {i for t in range(len(script)) for i in tracker.get_track_results()["id"].tolist()}
    assert len(tracker.get_track_results()["id"]) == 2
