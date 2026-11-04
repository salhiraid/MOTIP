#!/usr/bin/env python
"""Convert a DEIM(-D-FINE) checkpoint trained with the in-house mmdet framework (`DEIMDFINE`, `DFINEHead`, ...) into
the checkpoint format of `models/deim` (the official DEIM layout used by MOTIP2).

    # 1) look at both sides (send me this file if the conversion below reports anything unmatched)
    python tools/vehicles/convert_mmdet_deim.py --ckpt epoch_140.pth --size S --num-classes 5 --inspect inspect.txt
    # 2) convert
    python tools/vehicles/convert_mmdet_deim.py --ckpt epoch_140.pth --size S --num-classes 5 \
        --out pretrains/deim_s_vehicles_converted.pth

Why alignment instead of a fixed key table: the mmdet port may name modules differently from official DEIM
(`neck.convs.N.bn` vs `encoder.input_proj.N.norm`, `bbox_head.cls_branches` vs `decoder.dec_score_head`, ...), and I have
not seen your checkpoint. Both models are the same network (HGNetV2-B0, 3 decoder layers, hidden 256, reg_max 32,
num_points [3,6,3]), so every tensor of the target has exactly one source tensor of the same shape. Tensors are therefore
matched inside each shape bucket by (a) name similarity (numeric layer indices count, synonyms bn/norm etc.) and (b) the
relative order inside the bucket, solved optimally with the Hungarian algorithm. Anything ambiguous is listed in the report.
The output is refused unless every parameter of the target model is matched (override: --allow-partial).

ALWAYS validate the result functionally: detection-only evaluation of the converted weights on your validation split must
give detection quality close to what your mmdet model reports (see docs/VEHICLES.md, step 5).
"""
import argparse
import difflib
import os
import re
import sys
from collections import defaultdict

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

SYNONYMS = {"bn": "norm", "convs": "input_proj", "cls": "score", "class": "score", "classifier": "score",
            "reg": "bbox", "regression": "bbox", "branches": "head", "fc": "layers", "linear": "layers",
            "attn": "attn", "downsample": "downsample", "ffn": "ffn", "mlp": "layers"}
NUM_BATCHES = "num_batches_tracked"
# target tensors that may legitimately stay at their initial value
OPTIONAL_TARGET = ("num_batches_tracked", "anchors", "valid_mask", "pos_embed")
# source tensors that never exist in the official model (keypoint heads etc.)
DROP_SOURCE_HINTS = ("kp", "keypoint", "kpt", "oks", "vis", "mask_head", "momentum", "steps")


def tokens(key):
    out = []
    for t in re.split(r"[._]", key.lower()):
        if t:
            out.append(SYNONYMS.get(t, t))
    return out


def name_similarity(a, b):
    ta, tb = tokens(a), tokens(b)
    sa, sb = set(ta), set(tb)
    jac = len(sa & sb) / max(1, len(sa | sb))
    seq = difflib.SequenceMatcher(None, ta, tb).ratio()
    nums_a, nums_b = [t for t in ta if t.isdigit()], [t for t in tb if t.isdigit()]
    num_bonus = 0.3 if nums_a == nums_b and nums_a else (-0.3 if nums_a and nums_b and len(nums_a) == len(nums_b) else 0.0)
    tail = 0.1 if ta[-1:] == tb[-1:] else -0.5
    return 0.5 * jac + 0.5 * seq + num_bonus + tail


def load_source(path, which):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cands = {k: ck[k] for k in ("state_dict", "ema_state_dict", "model", "ema") if isinstance(ck.get(k), dict)}
    if not cands:
        cands = {"<root>": ck}
    print(f"checkpoint sections: {list(cands)} (top-level keys: {list(ck)[:10]})")
    name = which if which != "auto" else ("state_dict" if "state_dict" in cands else next(iter(cands)))
    sd = cands[name]
    if isinstance(sd, dict) and "module" in sd and isinstance(sd["module"], dict):
        sd = sd["module"]
    out = {}
    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        for pre in ("module.", "ema_model.", "model."):
            while k.startswith(pre):
                k = k[len(pre):]
        out[k] = v
    print(f"using section '{name}': {len(out)} tensors")
    return out


def build_target(size, num_classes, num_keypoints=0, keypoint_hidden_layers=4):
    from models.deim.builder import build_deim
    return build_deim(size, num_classes=num_classes, num_queries=300, num_keypoints=num_keypoints,
                      keypoint_hidden_layers=keypoint_hidden_layers)


# mmdet DFINEHead keypoint branches: `kps_xy_branches.0` / `kps_vis_branches.0` = nn.Sequential(Linear, ReLU, ..., Linear)
# (one branch shared by all decoder layers), mapped 1:1 onto models/deim/keypoints.py (`xy_branch` / `vis_branch`).
KP_PATTERNS = {"xy_branch": re.compile(r"(?:^|\.)kps_xy_branches\.0\.(\d+)\.(weight|bias)$"),
               "vis_branch": re.compile(r"(?:^|\.)kps_vis_branches\.0\.(\d+)\.(weight|bias)$")}


def split_keypoint_tensors(src):
    """-> (source without keypoint tensors, {target key: tensor}, K, hidden_layers). K = 0 when there is no keypoint head."""
    rest, kp, last = {}, {}, {}
    for k, v in src.items():
        for branch, pat in KP_PATTERNS.items():
            m = pat.search(k)
            if m:
                kp[f"keypoint_head.{branch}.{m.group(1)}.{m.group(2)}"] = v
                if m.group(2) == "weight":
                    last[branch] = max(last.get(branch, (-1, None)), (int(m.group(1)), v.shape[0]))
                break
        else:
            rest[k] = v
    if not kp:
        return src, {}, 0, 0
    assert set(last) == {"xy_branch", "vis_branch"}, f"incomplete keypoint branches in the source: {sorted(last)}"
    (ix, ox), (iv, ov) = last["xy_branch"], last["vis_branch"]
    assert ix == iv and ix % 2 == 0, f"xy/vis keypoint branches differ in depth ({ix} vs {iv})"
    assert ox == 2 * ov, f"xy branch has {ox} outputs but the visibility branch {ov} (expected 2x)"
    return rest, kp, ov, ix // 2


def slice_classes(src, src_classes, class_indices):
    """Select rows of the class-dependent tensors of a source with `src_classes` classes."""
    out = {}
    for k, v in src.items():
        kl = k.lower()
        classy = any(h in kl for h in ("cls", "score", "class", "label"))
        if classy and v.dim() >= 1 and v.shape[0] == src_classes:
            v = v[class_indices]
        elif classy and v.dim() >= 1 and v.shape[0] == src_classes + 1:
            v = torch.cat([v[class_indices], v[-1:]], 0)
        out[k] = v
    return out


def prefix(key, depth):
    return ".".join(key.split(".")[:depth])


def align(src, tgt, passes=3, rank_weight=0.25, prefix_weight=1.5):
    """Returns (mapping tgt_key -> (src_key, score, margin)), unmatched_src, unmatched_tgt.

    Pass 1 uses name similarity + a small in-bucket order tie-breaker. Passes 2.. learn, from the matches that were
    unambiguous (unique shape, or a clear score margin), which source module prefix corresponds to which target prefix
    (e.g. `neck.convs` -> `encoder.input_proj`) and add that as extra evidence for the harder (same-shape) tensors."""
    s_items = [(k, tuple(v.shape), v.dtype.is_floating_point) for k, v in src.items() if not k.endswith(NUM_BATCHES)
               and not any(h in k.lower() for h in DROP_SOURCE_HINTS)]
    t_items = [(k, tuple(v.shape), v.dtype.is_floating_point) for k, v in tgt.items() if not k.endswith(NUM_BATCHES)]
    sb, tb = defaultdict(list), defaultdict(list)
    for k, sh, fl in s_items:
        sb[(sh, fl)].append(k)
    for k, sh, fl in t_items:
        tb[(sh, fl)].append(k)

    votes = None
    for it in range(passes):
        mapping, un_t, used_src, confident = {}, [], set(), []
        for bucket, tkeys in tb.items():
            skeys = sb.get(bucket, [])
            if not skeys:
                un_t += tkeys
                continue
            n_s, n_t = len(skeys), len(tkeys)
            score = np.zeros((n_t, n_s))
            for a_, tk in enumerate(tkeys):
                for b_, sk in enumerate(skeys):
                    rank = 1.0 - abs(a_ / max(1, n_t - 1) - b_ / max(1, n_s - 1)) if n_t > 1 and n_s > 1 else 1.0
                    sc = name_similarity(sk, tk) + rank_weight * rank
                    if votes is not None:
                        sc += prefix_weight * votes.get((prefix(sk, 2), prefix(tk, 2)), 0.0) \
                            + 0.5 * prefix_weight * votes.get((prefix(sk, 3), prefix(tk, 3)), 0.0)
                    score[a_, b_] = sc
            rows, cols = linear_sum_assignment(-score)
            for a_, b_ in zip(rows, cols):
                second = np.sort(np.delete(score[a_], b_))[::-1]
                margin = float(score[a_, b_] - second[0]) if len(second) else float("inf")
                mapping[tkeys[a_]] = (skeys[b_], float(score[a_, b_]), margin)
                used_src.add(skeys[b_])
                if n_s == 1 or margin > 0.3:
                    confident.append((skeys[b_], tkeys[a_]))
            un_t += [tkeys[a_] for a_ in range(n_t) if a_ not in set(rows)]
        # learn prefix correspondences (normalised per source prefix) for the next pass
        votes = {}
        for depth in (2, 3):
            cnt, tot = defaultdict(float), defaultdict(float)
            for sk, tk in confident:
                cnt[(prefix(sk, depth), prefix(tk, depth))] += 1
                tot[prefix(sk, depth)] += 1
            votes.update({k: v / tot[k[0]] for k, v in cnt.items()})
    un_s = [k for k, _, _ in s_items if k not in used_src]
    return mapping, un_s, un_t


def group(keys, depth=2):
    g = defaultdict(int)
    for k in keys:
        g[".".join(k.split(".")[:depth])] += 1
    return dict(sorted(g.items()))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--which", default="auto", help="checkpoint section: auto | state_dict | ema_state_dict | ...")
    ap.add_argument("--size", default="S")
    ap.add_argument("--num-classes", type=int, default=5, help="classes of the TARGET (and normally the source) head")
    ap.add_argument("--src-num-classes", type=int, default=None, help="if the source head has a different class count")
    ap.add_argument("--class-indices", type=int, nargs="+", default=None, help="source rows -> target classes")
    ap.add_argument("--num-keypoints", type=int, default=None,
                    help="default: detected from the source (kps_xy_branches / kps_vis_branches); 0 drops them")
    ap.add_argument("--out", default=None)
    ap.add_argument("--inspect", default=None, help="write source/target key+shape listings to this file and stop")
    ap.add_argument("--allow-partial", action="store_true", help="save even if required target tensors are unmatched")
    ap.add_argument("--min-margin", type=float, default=0.05, help="below this score margin a match is called ambiguous")
    a = ap.parse_args(argv)

    src = load_source(a.ckpt, a.which)
    src, kp_state, K, L = split_keypoint_tensors(src)
    if a.num_keypoints is not None and a.num_keypoints != K:
        if a.num_keypoints == 0:
            kp_state, K = {}, 0
        else:
            raise SystemExit(f"--num-keypoints {a.num_keypoints} but the source head has {K}")
    if K:
        print(f"keypoint head found in the source: {K} keypoints, {L} hidden layers per branch")
    tmodel = build_target(a.size, a.num_classes, K, L or 4)
    tgt_full = tmodel.state_dict()
    tgt = {k: v for k, v in tgt_full.items() if not k.startswith("keypoint_head.")}      # aligned explicitly, not by similarity
    for k, v in kp_state.items():
        assert k in tgt_full and tgt_full[k].shape == v.shape, \
            f"keypoint tensor {k}: source {tuple(v.shape)} vs target {tuple(tgt_full[k].shape) if k in tgt_full else None} " \
            f"(the keypoint MLP must be hidden_dim wide = the DEIM decoder width)"

    if a.inspect:
        with open(a.inspect, "w") as f:
            f.write(f"# SOURCE {a.ckpt}: {len(src)} tensors\n")
            f.writelines(f"{k}\t{tuple(v.shape)}\t{str(v.dtype).replace('torch.', '')}\n" for k, v in src.items())
            f.write(f"\n# TARGET models.deim size={a.size} num_classes={a.num_classes}: {len(tgt)} tensors\n")
            f.writelines(f"{k}\t{tuple(v.shape)}\t{str(v.dtype).replace('torch.', '')}\n" for k, v in tgt.items())
        print(f"wrote {a.inspect}")
        return 0

    if a.src_num_classes and a.src_num_classes != a.num_classes:
        assert a.class_indices and len(a.class_indices) == a.num_classes, "--class-indices needs one source row per target class"
        src = slice_classes(src, a.src_num_classes, a.class_indices)

    mapping, un_s, un_t = align(src, tgt)
    required_missing = [k for k in un_t if not any(o in k for o in OPTIONAL_TARGET)]
    ambiguous = {t: m for t, m in mapping.items() if m[2] < a.min_margin}
    lines = [f"matched {len(mapping)} / {len([k for k in tgt if not k.endswith(NUM_BATCHES)])} target tensors",
             f"unmatched TARGET tensors needing a source: {len(required_missing)}",
             f"unmatched SOURCE tensors (dropped): {len(un_s)}   by prefix: {group(un_s)}",
             f"ambiguous matches (score margin < {a.min_margin}): {len(ambiguous)}", ""]
    lines += ["--- required target tensors without a match ---"] + [f"  {k}  {tuple(tgt[k].shape)}" for k in required_missing]
    lines += ["", "--- ambiguous matches (target <- source, score, margin) ---"]
    lines += [f"  {t} <- {s}  ({sc:.2f}, {mg:.3f})" for t, (s, sc, mg) in sorted(ambiguous.items())]
    lines += ["", "--- full mapping (target <- source, score, margin) ---"]
    lines += [f"  {t} <- {s}  ({sc:.2f}, {mg:.3f})" for t, (s, sc, mg) in sorted(mapping.items())]
    report = "\n".join(lines)
    rep_path = (a.out or a.ckpt) + ".convert_report.txt"
    open(rep_path, "w").write(report)
    print("\n".join(lines[:5]))
    print(f"full report: {rep_path}")

    if required_missing and not a.allow_partial:
        print("REFUSING to write the checkpoint: required tensors are unmatched (see report). Send me the --inspect file.")
        return 2

    new_state = dict(tgt_full)                             # unmatched optional tensors keep their initial values
    new_state.update(kp_state)                             # converted keypoint head (empty without one)
    for t, (s, _, _) in mapping.items():
        v = src[s]
        new_state[t] = v.to(tgt[t].dtype) if v.shape == tgt[t].shape else tgt[t]
    for k in ("decoder.up", "decoder.reg_scale"):         # fixed constants of D-FINE: check they survived
        if k in mapping and not torch.allclose(new_state[k].float(), tgt[k].float()):
            print(f"WARNING {k}: converted value {new_state[k].tolist()} != expected {tgt[k].tolist()}")
    if a.out:
        from models.deim.wrapper import load_deim_pretrain
        check = build_target(a.size, a.num_classes, K, L or 4)
        load_deim_pretrain(check, state=new_state)         # strict load round-trip
        check.eval()
        with torch.no_grad():
            o = check(torch.rand(1, 3, 224, 320))
        assert torch.isfinite(o["pred_logits"]).all() and torch.isfinite(o["pred_boxes"]).all()
        if K:
            assert o["pred_keypoints"].shape[-2:] == (K, 2) and torch.isfinite(o["pred_keypoints"]).all()
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        torch.save({"model": new_state, "meta": {"converted_from": a.ckpt, "size": a.size, "num_classes": a.num_classes,
                                                 "num_keypoints": K, "keypoint_hidden_layers": L,
                                                 "ambiguous": len(ambiguous)}}, a.out)
        print(f"saved {a.out} (strict load + forward OK; {len(ambiguous)} ambiguous matches -> run the detection-only check)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
