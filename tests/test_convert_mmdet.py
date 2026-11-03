# Converter for the user's mmdet-trained DEIM checkpoint: tested on SYNTHETIC mmdet-style checkpoints (real key names are unknown).
import os
import re
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools", "vehicles"))

import convert_mmdet_deim as C
from models.deim.builder import build_deim
from models.deim.wrapper import load_deim_pretrain


def make_target(seed=0, size="S"):
    torch.manual_seed(seed)
    m = build_deim(size, num_classes=5)
    for p in m.parameters():                      # distinct random values everywhere => exact-recovery test
        if p.requires_grad:
            torch.nn.init.normal_(p, std=0.02)
    for name, b in m.named_buffers():
        if name.endswith("running_var"):
            b.copy_(1.0 + torch.rand_like(b))           # valid BN statistics (variance > 0)
        elif name.endswith("running_mean"):
            b.copy_(0.1 * torch.randn_like(b))
    return m


def to_mmdet_style(state):
    """Plausible renaming into an mmdet-flavoured layout (names are a GUESS; the real ones differ)."""
    out = {}
    for k, v in state.items():
        n = k
        n = re.sub(r"^encoder\.input_proj\.(\d+)\.conv\.", r"neck.convs.\1.conv.", n)
        n = re.sub(r"^encoder\.input_proj\.(\d+)\.norm\.", r"neck.convs.\1.bn.", n)
        n = re.sub(r"^decoder\.dec_score_head\.(\d+)\.", r"bbox_head.cls_branches.\1.", n)
        n = re.sub(r"^decoder\.dec_bbox_head\.(\d+)\.", r"bbox_head.reg_branches.\1.", n)
        n = n.replace("decoder.enc_score_head.", "bbox_head.enc_cls_head.").replace("decoder.enc_bbox_head.", "bbox_head.enc_reg_head.")
        n = n.replace("decoder.decoder.layers.", "decoder.layers.").replace("decoder.decoder.lqe_layers.", "bbox_head.lqe_layers.")
        n = n.replace("decoder.pre_bbox_head.", "bbox_head.pre_reg_head.")
        out["module." + n] = v.clone()
    return out


def add_extras(sd):
    sd = dict(sd)
    for i in range(3):                              # keypoint heads that do not exist in the official model
        sd[f"module.bbox_head.kp_branches.{i}.layers.0.weight"] = torch.randn(256, 256)
        sd[f"module.bbox_head.kp_branches.{i}.layers.0.bias"] = torch.randn(256)
        sd[f"module.bbox_head.kp_vis_branches.{i}.weight"] = torch.randn(31, 256)
    return sd


def same_weights(model_a, model_b):
    a, b = model_a.state_dict(), model_b.state_dict()
    bad = [k for k in a if "num_batches_tracked" not in k and not torch.equal(a[k], b[k])
           and not any(o in k for o in ("anchors", "valid_mask", "pos_embed"))]
    return bad


def run(tmp_path, src_state, section="state_dict", extra_args=()):
    ck = tmp_path / "src.pth"
    torch.save({section: src_state, "meta": {"epoch": 140}}, ck)
    out = tmp_path / "conv.pth"
    rc = C.main(["--ckpt", str(ck), "--size", "S", "--num-classes", "5", "--out", str(out), *extra_args])
    return rc, out


def test_recovers_exact_weights_from_renamed_scrambled_checkpoint_with_extra_heads(tmp_path):
    target = make_target()
    src = add_extras(to_mmdet_style(target.state_dict()))
    src = {k: src[k] for k in sorted(src)}                         # scramble the registration order
    rc, out = run(tmp_path, src)
    assert rc == 0
    loaded = build_deim("S", num_classes=5)
    load_deim_pretrain(loaded, str(out))
    assert same_weights(loaded, target) == []                       # EVERY tensor routed to the right place


def test_recovers_from_opaque_names_using_order_only(tmp_path):
    target = make_target(seed=1)
    names = list(target.state_dict())
    src = {f"p{i:05d}": v.clone() for i, (k, v) in enumerate(target.state_dict().items())}   # names carry NO information
    rc, out = run(tmp_path, src)
    assert rc == 0
    loaded = build_deim("S", num_classes=5)
    load_deim_pretrain(loaded, str(out))
    assert same_weights(loaded, target) == []


def test_ema_section_selection_and_refusal_when_a_required_tensor_is_missing(tmp_path):
    target = make_target(seed=2)
    src = to_mmdet_style(target.state_dict())
    rc, out = run(tmp_path, src, section="ema_state_dict", extra_args=["--which", "ema_state_dict"])
    assert rc == 0
    broken = {k: v for k, v in src.items() if "decoder.layers.1.cross_attn.sampling_offsets" not in k}
    ck = tmp_path / "broken.pth"
    torch.save({"state_dict": broken}, ck)
    out3 = tmp_path / "broken_out.pth"
    assert C.main(["--ckpt", str(ck), "--size", "S", "--num-classes", "5", "--out", str(out3)]) == 2
    assert not out3.exists()                                                       # refuses to write a half-converted model
    assert "required" in open(str(out3) + ".convert_report.txt").read()


def test_class_head_row_selection_when_source_has_more_classes(tmp_path):
    target = make_target(seed=3)
    st = target.state_dict()
    big = {}
    for k, v in st.items():                                         # pretend the source head had 8 classes (rows 1,2,3,5,7 = ours)
        if "score_head" in k and v.shape[0] == 5:
            extra = torch.randn(3, *v.shape[1:])
            v = torch.cat([extra[:1], v[:3], extra[1:2], v[3:4], extra[2:3], v[4:5]], 0) if v.dim() == 2 else \
                torch.cat([extra[:1], v[:3], extra[1:2], v[3:4], extra[2:3], v[4:5]], 0)
        elif k.endswith("denoising_class_embed.weight"):
            extra = torch.randn(3, v.shape[1])
            v = torch.cat([extra[:1], v[:3], extra[1:2], v[3:4], extra[2:3], v[4:5], v[5:]], 0)
        big[k] = v
    rc, out = run(tmp_path, to_mmdet_style(big), extra_args=["--src-num-classes", "8", "--class-indices", "1", "2", "3", "5", "7"])
    assert rc == 0
    loaded = build_deim("S", num_classes=5)
    load_deim_pretrain(loaded, str(out))
    assert same_weights(loaded, target) == []


def test_inspect_mode_lists_both_sides(tmp_path):
    ck = tmp_path / "s.pth"
    torch.save({"state_dict": add_extras(to_mmdet_style(make_target().state_dict()))}, ck)
    txt = tmp_path / "inspect.txt"
    assert C.main(["--ckpt", str(ck), "--size", "S", "--inspect", str(txt)]) == 0
    t = txt.read_text()
    assert "# SOURCE" in t and "# TARGET" in t and "bbox_head.cls_branches.0.weight" in t and "decoder.dec_score_head.0.weight" in t
