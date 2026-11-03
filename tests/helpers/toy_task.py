"""Toy association task on DETECTION-FEATURE level (no images): K identities are assigned randomly per clip, objects move
smoothly, and the ID decoder must tell which identity each detection of frame t belongs to using the labelled history.
Each knob isolates one mechanism of the paper so that it can be switched off and the failure observed:
  content      'distinct'  -> objects look different (appearance is enough)
               'identical' -> every object has the SAME appearance: only position/motion tells them apart (Spatial Anchor)
  stale_labels True -> history older than one frame carries consistently WRONG identity labels: only the most recent frame
                       can be trusted -> requires a working temporal encoding (RoPE / learned bias). Query = last frame only.
"""
import math

import torch

from models.motip.id_decoder import IDDecoder


def fourier_pos(xy, dim, seed=123, scale=3.0):
    """Stand-in for the detector's positional MLP p_i = query_pos_head(box): fixed random Fourier features of the box centre."""
    g = torch.Generator().manual_seed(seed)
    B = torch.randn(2, dim // 2, generator=g) * scale
    proj = 2 * math.pi * xy @ B
    return torch.cat([proj.sin(), proj.cos()], -1)


def make_batch(B, T=7, N=4, D=48, K=10, content="distinct", stale_labels=False, seed=None, noise=0.05, with_pos=True):
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    r = lambda *s: torch.randn(*s, generator=g)
    # identities: a random subset of K labels per clip (MOTIP: labels are arbitrary and only meaningful inside the clip)
    labels = torch.stack([torch.randperm(K, generator=g)[:N] for _ in range(B)])                  # (B, N)
    base = r(B, N, D) if content == "distinct" else r(B, 1, D).expand(B, N, D).clone()
    # well separated start positions on a grid + small smooth motion
    grid = torch.tensor([[0.2, 0.2], [0.2, 0.8], [0.8, 0.2], [0.8, 0.8], [0.5, 0.5], [0.5, 0.2]])[:N]
    perm = torch.stack([torch.randperm(N, generator=g) for _ in range(B)])
    start = grid[perm] + 0.03 * r(B, N, 2)
    vel = 0.015 * r(B, N, 2)
    t = torch.arange(T).view(1, T, 1, 1)
    pos_xy = start[:, None] + vel[:, None] * t                                                      # (B, T, N, 2)
    feat = base[:, None] + noise * r(B, T, N, D)
    pe = fourier_pos(pos_xy, D) if with_pos else None
    ids = labels[:, None, :].expand(B, T, N).clone()
    hist_ids = ids.clone()
    if stale_labels:
        # Frames older than T-2 carry a CONSISTENT BUT WRONG identity assignment (labels cyclically shifted); frame T-2 is
        # correct. Majority voting over the context is therefore wrong; only a decoder that knows which context frame is
        # "one step ago" (temporal encoding) can answer the query at the last frame correctly.
        hist_ids[:, : T - 2] = torch.roll(ids[:, : T - 2], shifts=1, dims=-1)
    si = {
        "trajectory_features": feat[:, None], "unknown_features": feat[:, None],
        "trajectory_id_labels": hist_ids[:, None], "unknown_id_labels": ids[:, None],
        "trajectory_times": torch.arange(T).view(1, 1, T, 1).expand(B, 1, T, N).clone(),
        "unknown_times": torch.arange(T).view(1, 1, T, 1).expand(B, 1, T, N).clone(),
        "trajectory_masks": torch.zeros(B, 1, T, N, dtype=torch.bool), "unknown_masks": torch.zeros(B, 1, T, N, dtype=torch.bool),
    }
    if with_pos:
        si["trajectory_pos"], si["unknown_pos"] = pe[:, None], pe[:, None]
    return si


def make_decoder(D=48, K=10, layers=2, seed=0, **kw):
    torch.manual_seed(seed)                       # deterministic initialisation (learning outcomes can depend on it)
    kw.setdefault("id_fusion", "add")
    kw.setdefault("pos_enc", "rope")
    return IDDecoder(feature_dim=D, id_dim=D, ffn_dim_ratio=4, num_layers=layers, head_dim=24, num_id_vocabulary=K,
                     rel_pe_length=16, use_aux_loss=False, use_shared_aux_head=True, **kw)


def train_eval(decoder, make_train, make_eval, steps=250, lr=2e-3, B=24, seed=0, query_from=1):
    """CE on frames >= query_from; returns accuracy on a held-out batch (identity of every unknown token)."""
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(decoder.parameters(), lr=lr, weight_decay=0.0)
    decoder.train()
    for it in range(steps):
        si = make_train(B, 1000 + it)
        logits, labels, masks = decoder(si, False)
        loss = torch.nn.functional.cross_entropy(logits[:, :, query_from:].reshape(-1, logits.shape[-1]),
                                                 labels[:, :, query_from:].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
        opt.step()
    decoder.eval()
    with torch.no_grad():
        si = make_eval(256, 777)
        logits, _, masks = decoder(si, False)
        pred = logits[:, :, query_from:].argmax(-1)
        return float((pred == si["unknown_id_labels"][:, :, query_from:]).float().mean()), float(loss)


def world(T=40, N=4, D=48, seed=0, content="distinct", vel=0.004, drop=None):
    """A video of N smoothly moving objects as the tracker would see it: per frame the detections (appearance feature,
    positional encoding p_i, normalised box). Returns feat (T,N,D), xy (T,N,2). Used to run the REAL RuntimeTracker with an
    ID decoder trained on `make_batch` data (same feature conventions)."""
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(N, D, generator=g) if content == "distinct" else torch.randn(1, D, generator=g).expand(N, D).clone()
    grid = torch.tensor([[0.2, 0.2], [0.2, 0.8], [0.8, 0.2], [0.8, 0.8], [0.5, 0.5], [0.5, 0.2]])[:N]
    start = grid + 0.03 * torch.randn(N, 2, generator=g)
    v = vel * torch.randn(N, 2, generator=g)
    xy = start[None] + v[None] * torch.arange(T).view(T, 1, 1)
    feat = base[None] + 0.05 * torch.randn(T, N, D, generator=g)
    return feat, xy
