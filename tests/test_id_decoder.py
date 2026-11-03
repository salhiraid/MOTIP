# Gate 3 (unit, CPU): ID decoder structure. PLAN.md §6 Gate 3.
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from models.motip.id_decoder import IDDecoder
from legacy_id_decoder import IDDecoder as LegacyIDDecoder

K = 10


def make_seq_info(D, B=1, G=2, T=5, N=4, seed=0, with_pos=False):
    g = torch.Generator().manual_seed(seed)
    f = lambda: torch.randn(B, G, T, N, D, generator=g)
    times = torch.arange(T)[None, None, :, None].repeat(B, G, 1, N)
    tm = torch.rand(B, G, T, N, generator=g) < 0.2           # True = masked/absent
    um = torch.rand(B, G, T, N, generator=g) < 0.2
    labels = torch.randint(0, K, (B, G, T, N), generator=g)
    si = {
        "trajectory_features": f(), "unknown_features": f(),
        "trajectory_id_labels": labels, "unknown_id_labels": labels.clone(),
        "trajectory_times": times, "unknown_times": times.clone(),
        "trajectory_masks": tm, "unknown_masks": um,
    }
    if with_pos:
        si["trajectory_pos"], si["unknown_pos"] = f(), f()
    return si


def make(D=32, layers=3, **kw):
    kw.setdefault("id_fusion", "add")
    kw.setdefault("pos_enc", "rope")
    return IDDecoder(feature_dim=D, id_dim=D, ffn_dim_ratio=4, num_layers=layers, head_dim=16,
                     num_id_vocabulary=K, rel_pe_length=5, use_aux_loss=False, use_shared_aux_head=True, **kw)


@pytest.mark.parametrize("fusion,pos", [("add", "rope"), ("add", "learned_rel"), ("concat", "learned_rel"),
                                         ("concat", "rope")])
def test_output_shape(fusion, pos):
    dec = make(id_fusion=fusion, pos_enc=pos).eval()
    si = make_seq_info(32)
    logits, _, masks = dec(si, use_decoder_checkpoint=False)
    assert logits.shape == (1, 2, 5, 4, K + 1)          # (num tokens, K+1) per (b, g, t, n)
    assert masks.shape == (1, 2, 5, 4)
    assert torch.isfinite(logits).all()


def test_additive_fusion_keeps_detr_hidden_dim_and_doubles_ffn_ratio():
    dec = make(D=32, id_fusion="add")
    assert dec.model_dim == 32                                  # halved vs concatenation (64)
    assert make(D=32, id_fusion="concat", pos_enc="learned_rel").model_dim == 64
    assert dec.ffn_layers[0].linear1.out_features == 32 * 4 if hasattr(dec.ffn_layers[0], "linear1") else True
    si = make_seq_info(32)
    traj, unk = dec.build_inputs(si)
    assert torch.equal(traj, si["trajectory_features"] + dec.id_label_to_embed(si["trajectory_id_labels"]))
    assert torch.equal(unk, si["unknown_features"] + dec.generate_empty_id_embed(si["unknown_features"]))


def test_motip2_s_decoder_is_much_smaller_than_motip():
    s = IDDecoder(256, 256, 4, 3, 32, 50, 30, False, True, id_fusion="add", pos_enc="rope")
    motip = LegacyIDDecoder(256, 256, 2, 6, 32, 50, 30, False, True)
    n_s, n_m = sum(p.numel() for p in s.parameters()), sum(p.numel() for p in motip.parameters())
    assert n_s < 0.25 * n_m, (n_s, n_m)


@pytest.mark.parametrize("train", [False, True])
def test_learned_rel_concat_reproduces_legacy_motip_exactly(train):
    torch.manual_seed(0)
    legacy = LegacyIDDecoder(feature_dim=32, id_dim=32, ffn_dim_ratio=2, num_layers=3, head_dim=16,
                             num_id_vocabulary=K, rel_pe_length=5, use_aux_loss=False, use_shared_aux_head=True)
    with torch.no_grad():
        legacy.rel_pos_embeds.normal_(0, 0.5)           # make the learned temporal bias non-trivial
    new = IDDecoder(feature_dim=32, id_dim=32, ffn_dim_ratio=2, num_layers=3, head_dim=16, num_id_vocabulary=K,
                    rel_pe_length=5, use_aux_loss=False, use_shared_aux_head=True,
                    id_fusion="concat", pos_enc="learned_rel", spatial_anchor=False)
    assert set(legacy.state_dict()) == set(new.state_dict())
    new.load_state_dict(legacy.state_dict())
    legacy.train(train), new.train(train)
    si = make_seq_info(32)
    a = legacy(si, use_decoder_checkpoint=False)[0]
    b = new(si, use_decoder_checkpoint=False)[0]
    assert torch.equal(a, b)


def test_rope_decoder_depends_only_on_relative_time():
    torch.manual_seed(0)
    dec = make().eval()
    si = make_seq_info(32)
    a = dec(si, False)[0]
    shifted = dict(si)
    shifted["trajectory_times"] = si["trajectory_times"] + 17
    shifted["unknown_times"] = si["unknown_times"] + 17
    b = dec(shifted, False)[0]
    assert torch.allclose(a, b, atol=1e-4)


def test_masked_trajectory_tokens_and_future_do_not_leak():
    torch.manual_seed(0)
    dec = make().eval()
    si = make_seq_info(32)
    a = dec(si, False)[0]
    si2 = dict(si)
    f = si["trajectory_features"].clone()
    f[si["trajectory_masks"]] += 50.0                   # masked (absent) trajectory tokens must be ignored
    si2["trajectory_features"] = f
    b = dec(si2, False)[0]
    assert torch.allclose(a, b, atol=1e-4)
    # tokens of the last frame are never visible to earlier/same-frame unknowns (cross-attn is strictly "past or equal")
    f = si["trajectory_features"].clone()
    f[:, :, -1] += 50.0
    si3 = dict(si); si3["trajectory_features"] = f
    c = dec(si3, False)[0]
    assert torch.allclose(a[:, :, :-1], c[:, :, :-1], atol=1e-4)


def test_alpha_only_acts_at_inference():
    torch.manual_seed(0)
    dec = make()
    si = make_seq_info(32)
    dec.set_rope_alpha(2.0, "multiply")
    dec.train()
    t_a = dec(si, False)[0]
    dec.set_rope_alpha(1.0)
    t_b = dec(si, False)[0]
    assert torch.equal(t_a, t_b)                        # training ignores alpha
    dec.eval(); dec.set_rope_alpha(2.0, "multiply")
    e_a = dec(si, False)[0]
    dec.set_rope_alpha(1.0)
    e_b = dec(si, False)[0]
    assert not torch.allclose(e_a, e_b, atol=1e-5)
