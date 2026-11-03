# Gate 2 (unit, CPU): DEIM wrapper contract. See PLAN.md §6 Gate 2 and §2 (p_i location).
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.deim.builder import build_deim, build_detection_criterion, deim_hidden_dim, deim_num_layers
from models.deim.wrapper import load_deim_pretrain

README_PARAMS_M = {"N": 4, "S": 10, "M": 19, "L": 31, "X": 62}   # DEIM README (80-class, deploy)


def _randomize_box_refinement(m, std=0.05):
    """DEIM zero-initialises its box-refinement heads (final box == layer-input box at init); randomise them so the
    difference between 'embedding of the box entering the last layer' and 'embedding of the final box' is visible."""
    for h in m.decoder.dec_bbox_head:
        torch.nn.init.normal_(h.layers[-1].weight, std=std)


def _targets(n_img, n_obj=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n_img):
        c = torch.rand(n_obj, 2, generator=g) * 0.5 + 0.25
        wh = torch.rand(n_obj, 2, generator=g) * 0.15 + 0.05
        out.append({"labels": torch.zeros(n_obj, dtype=torch.int64), "boxes": torch.cat([c, wh], -1)})
    return out


@pytest.mark.parametrize("size", ["N", "S", "M", "L", "X"])
def test_param_count_matches_deim_readme(size):
    m = build_deim(size, num_classes=80)       # the README counts the 80-class COCO model
    n = sum(p.numel() for p in m.parameters()) / 1e6
    ref = README_PARAMS_M[size]
    assert abs(n - ref) / ref < 0.2, f"{size}: {n:.1f}M params vs README {ref}M"


def test_id_decoder_mirror_dims():
    assert [deim_hidden_dim(s) for s in "NSMLX"] == [128, 256, 256, 256, 256]
    assert [deim_num_layers(s) for s in "NSMLX"] == [3, 3, 4, 6, 6]


def test_eval_outputs_and_p_i_is_last_layer_input_embedding():
    torch.manual_seed(0)
    m = build_deim("S", num_classes=1).eval()
    _randomize_box_refinement(m)
    calls = []
    m.decoder.query_pos_head.register_forward_hook(lambda mod, inp, out: calls.append((inp[0], out)))
    x = torch.rand(2, 3, 128, 192)
    with torch.no_grad():
        out = m(x)
    Q, D = 300, m.hidden_dim
    assert out["pred_logits"].shape == (2, Q, 1)
    assert out["pred_boxes"].shape == (2, Q, 4)
    assert out["outputs"].shape == (2, Q, D)          # e_i
    assert out["pos_embeds"].shape == (2, Q, D)       # p_i
    # one positional-MLP call per decoder layer up to eval_idx (S: 3 layers) -- check before we call the MLP ourselves
    assert len(calls) == deim_num_layers("S")
    # N1(a): p_i is the positional MLP output for the box ENTERING the last evaluated layer.
    box_in, pos = calls[-1]
    assert torch.equal(out["pos_embeds"], pos.clamp(-10, 10))
    # ...which is not the embedding of the final output box.
    final_pos = m.decoder.box_to_pos_embed(out["pred_boxes"])
    assert not torch.allclose(out["pos_embeds"], final_pos)


def test_final_box_anchor_source():
    torch.manual_seed(0)
    m = build_deim("S", num_classes=1, anchor_source="final_box").eval()
    _randomize_box_refinement(m)
    with torch.no_grad():
        out = m(torch.rand(1, 3, 128, 192))
        assert torch.equal(out["pos_embeds"], m.decoder.box_to_pos_embed(out["pred_boxes"]))


def test_lightweight_equals_full_in_eval_and_dynamic_sizes():
    torch.manual_seed(0)
    m = build_deim("S", num_classes=1).eval()
    for hw in [(128, 192), (160, 288)]:                 # dynamic, non-square input sizes
        x = torch.rand(2, 3, *hw)
        with torch.no_grad():
            full = m(x)
            light = m(x, lightweight=True)
        assert set(light) == {"pred_logits", "pred_boxes", "outputs", "pos_embeds"}
        for k in light:
            assert torch.allclose(full[k], light[k], atol=1e-5), k


def test_train_forward_with_denoising_and_criterion_indices():
    torch.manual_seed(0)
    m = build_deim("S", num_classes=1).train()
    _randomize_box_refinement(m)       # DDF is skipped when all layers' corners are identical (zero-init heads)
    crit = build_detection_criterion(1)
    x = torch.rand(2, 3, 128, 192)
    tg = _targets(2)
    out = m(x, targets=tg)
    # dn queries must be split off: e_i / p_i only for the 300 regular queries
    assert out["outputs"].shape[:2] == (2, 300) and out["pos_embeds"].shape[:2] == (2, 300)
    losses, indices = crit(out, tg)
    # final layer: MAL + L1 + GIoU + FGL; DDF (self-distillation from the last layer) lives on the auxiliary layers
    assert {"loss_mal", "loss_bbox", "loss_giou", "loss_fgl"} <= set(losses)
    assert any(k.startswith("loss_ddf") for k in losses)
    assert len(indices) == 2 and all(len(i[0]) == len(i[1]) == 3 for i in indices)
    # matching-only path gives the same final-layer indices
    idx2 = crit.match(out, tg)
    assert all(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) for a, b in zip(indices, idx2))
    crit.total(losses).backward()
    assert m.decoder.query_pos_head.layers[0].weight.grad is not None


def test_lightweight_skips_denoising_in_train_mode():
    m = build_deim("S", num_classes=1).train()
    nd = m.decoder.num_denoising
    out = m(torch.rand(2, 3, 128, 192), lightweight=True)
    assert out["outputs"].shape == (2, 300, 256)
    assert m.decoder.num_denoising == nd                # restored


def test_coco_checkpoint_loading_slices_person_head(tmp_path):
    torch.manual_seed(0)
    coco = build_deim("S", num_classes=80)
    state = dict(coco.state_dict())
    state["decoder.anchors"] = torch.zeros(1, 10, 4)     # upstream fixed-size buffers must be tolerated
    state["decoder.valid_mask"] = torch.ones(1, 10, 1)
    path = tmp_path / "coco.pth"
    torch.save({"ema": {"module": state}}, path)
    ours = build_deim("S", num_classes=1)
    dropped = load_deim_pretrain(ours, str(path))
    assert set(dropped) == {"decoder.anchors", "decoder.valid_mask"}
    assert torch.equal(ours.decoder.enc_score_head.weight, coco.decoder.enc_score_head.weight[:1])
    assert torch.equal(ours.decoder.dec_score_head[-1].bias, coco.decoder.dec_score_head[-1].bias[:1])
    w = coco.decoder.denoising_class_embed.weight
    assert torch.equal(ours.decoder.denoising_class_embed.weight, torch.cat([w[:1], w[-1:]]))
    assert torch.equal(ours.backbone.stem.stem1.conv.weight, coco.backbone.stem.stem1.conv.weight)


def test_coco80_to_vehicle_classes_and_5_to_5_passthrough(tmp_path):
    """COCO-80 head -> ('bicycle','car','motorcycle','bus','truck') = rows [1,2,3,5,7]; a 5-class checkpoint loads as is."""
    torch.manual_seed(0)
    coco = build_deim("S", num_classes=80)
    path = tmp_path / "coco.pth"
    torch.save({"ema": {"module": coco.state_dict()}}, path)
    veh = build_deim("S", num_classes=5)
    load_deim_pretrain(veh, str(path), class_indices=[1, 2, 3, 5, 7])
    idx = [1, 2, 3, 5, 7]
    assert torch.equal(veh.decoder.enc_score_head.weight, coco.decoder.enc_score_head.weight[idx])
    for a, b in zip(veh.decoder.dec_score_head, coco.decoder.dec_score_head):
        assert torch.equal(a.bias, b.bias[idx])
    w = coco.decoder.denoising_class_embed.weight
    assert torch.equal(veh.decoder.denoising_class_embed.weight, torch.cat([w[idx], w[-1:]]))
    assert torch.equal(veh.decoder.query_pos_head.layers[0].weight, coco.decoder.query_pos_head.layers[0].weight)
    # round trip: a 5-class checkpoint (e.g. converted custom detector or a stage-1 checkpoint) needs no class_indices
    path2 = tmp_path / "veh.pth"
    torch.save({"model": veh.state_dict()}, path2)
    veh2 = build_deim("S", num_classes=5)
    load_deim_pretrain(veh2, str(path2))
    assert all(torch.equal(a, b) for a, b in zip(veh.state_dict().values(), veh2.state_dict().values()))
    # wrong number of indices is refused rather than silently mis-loaded
    with pytest.raises(AssertionError):
        load_deim_pretrain(build_deim("S", num_classes=5), str(path), class_indices=[1, 2, 3])
