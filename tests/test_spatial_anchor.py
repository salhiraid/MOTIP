# Gate 4 (unit, CPU): Spatial Anchor. PLAN.md §6 Gate 4, decision #9 (added after fusion, before layer 0).
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from test_id_decoder import make, make_seq_info
from models.deim.builder import build_deim
from models.motip.id_decoder import IDDecoder


def _pair(seed=0):
    torch.manual_seed(seed)
    off = make(spatial_anchor=False)
    torch.manual_seed(seed)
    on = make(spatial_anchor=True)
    on.load_state_dict(off.state_dict())
    return off, on


def test_anchor_off_equals_phase3_input_bit_for_bit():
    off, on = _pair()
    si = make_seq_info(32, with_pos=True)
    t_off, u_off = off.build_inputs(si)
    # Phase-3 input, written out by hand: e_i + id_embedding
    assert torch.equal(t_off, si["trajectory_features"] + off.id_label_to_embed(si["trajectory_id_labels"]))
    assert torch.equal(u_off, si["unknown_features"] + off.generate_empty_id_embed(si["unknown_features"]))
    # pos present in seq_info must be ignored when the flag is off
    si_no_pos = {k: v for k, v in si.items() if "pos" not in k}
    t2, u2 = off.build_inputs(si_no_pos)
    assert torch.equal(t_off, t2) and torch.equal(u_off, u2)


def test_anchor_on_adds_exactly_p_i_to_trajectory_and_unknown_tokens():
    off, on = _pair()
    si = make_seq_info(32, with_pos=True)
    t_off, u_off = off.build_inputs(si)
    t_on, u_on = on.build_inputs(si)
    assert torch.equal(t_on, t_off + si["trajectory_pos"])
    assert torch.equal(u_on, u_off + si["unknown_pos"])


def test_anchor_adds_no_parameters():
    off, on = _pair()
    assert list(off.state_dict()) == list(on.state_dict())
    assert sum(p.numel() for p in off.parameters()) == sum(p.numel() for p in on.parameters())


def test_anchor_needs_pos_and_additive_fusion():
    _, on = _pair()
    with pytest.raises(AssertionError):
        on.build_inputs(make_seq_info(32, with_pos=False))
    with pytest.raises(AssertionError):
        make(id_fusion="concat", pos_enc="learned_rel", spatial_anchor=True)


def test_anchor_changes_decoder_output_and_is_trainable():
    off, on = _pair()
    off.eval(); on.eval()
    si = make_seq_info(32, with_pos=True)
    assert not torch.allclose(off(si, False)[0], on(si, False)[0], atol=1e-5)
    on.train()
    si_g = {k: (v.clone().requires_grad_(True) if k.endswith("_pos") else v) for k, v in si.items()}
    on(si_g, False)[0].sum().backward()
    assert si_g["trajectory_pos"].grad is not None and si_g["unknown_pos"].grad is not None  # grad reaches p_i


@pytest.mark.parametrize("size,hidden", [("N", 128), ("S", 256)])
def test_detector_p_i_dim_matches_decoder_width(size, hidden):
    det = build_deim(size, num_classes=1, num_queries=100).eval()
    with torch.no_grad():
        out = det(torch.rand(1, 3, 128, 192))
    assert out["pos_embeds"].shape[-1] == out["outputs"].shape[-1] == hidden
    dec = IDDecoder(hidden, hidden, 4, 3, 32, 10, 5, False, True, id_fusion="add", pos_enc="rope", spatial_anchor=True)
    assert dec.model_dim == out["pos_embeds"].shape[-1]
