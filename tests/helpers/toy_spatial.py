"""Toy task for the Spatial ID Loss: 6 slot positions (grid), random identity labels per clip, identical appearance, noisy
observed positions (-> the decoder sometimes confuses neighbours). Provides the boxes the loss needs."""
import torch

from helpers.toy_task import make_batch, fourier_pos

N, T, D, K = 6, 6, 48, 10
GRID = torch.tensor([[0.2, 0.2], [0.2, 0.8], [0.8, 0.2], [0.8, 0.8], [0.5, 0.5], [0.5, 0.2]])


def batch(B, seed, pos_noise):
    g = torch.Generator().manual_seed(seed)
    si = make_batch(B, T=T, N=N, D=D, K=K, content="identical", seed=seed, with_pos=False)
    xy = GRID[None, None].expand(B, T, N, 2) + 0.02 * torch.randn(B, T, N, 2, generator=g)
    pe = fourier_pos(xy + pos_noise * torch.randn(B, T, N, 2, generator=g), D)       # what the detector's p_i sees (noisy)
    si["trajectory_pos"], si["unknown_pos"] = pe[:, None], pe[:, None]
    box = torch.cat([xy, torch.full_like(xy, 0.18)], -1)
    extra = {"pred_boxes": (box + 0.005 * torch.randn_like(box))[:, None], "gt_boxes": box[:, None],
             "gt_present": torch.ones(B, 1, T, N, dtype=torch.bool), "real_id_labels": si["unknown_id_labels"].clone()}
    return si, extra
