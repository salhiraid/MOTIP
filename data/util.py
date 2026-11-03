# Copyright (c) Ruopeng Gao. All Rights Reserved.

import random
import torch
import einops
import torch.nn.functional as F

from utils.nested_tensor import nested_tensor_from_tensor_list


def is_legal(annotation: dict):
    assert "id" in annotation, "Annotation must have 'id' field."
    assert "category" in annotation, "Annotation must have 'category' field."
    assert "bbox" in annotation, "Annotation must have 'bbox' field."
    assert "visibility" in annotation, "Annotation must have 'visibility' field."

    assert len(annotation["id"]) == len(annotation["category"]) \
           == len(annotation["bbox"]) == len(annotation["visibility"]), \
           "The length of 'id', 'category', 'bbox', 'visibility' must be the same."

    # assert torch.unique(annotation["id"]).size(0) == annotation["id"].size(0), f"IDs must be unique."
    _id_unique = torch.unique(annotation["id"]).size(0) == annotation["id"].size(0)     # for PersonPath22

    # A hack implementation for DETR (300 queries):
    # TODO: to make it more general, maybe pass the number of queries as an parameter.
    leq_300 = annotation["id"].shape[0] <= 300

    # return len(annotation["id"]) > 0
    return len(annotation["id"]) > 0 and _id_unique and leq_300


def append_annotation(
        annotation: dict,
        obj_id: int,
        category: int,
        bbox: list,
        visibility: float,
):
    annotation["id"] = torch.cat([
        annotation["id"],
        torch.tensor([obj_id], dtype=torch.int64)
    ])
    annotation["category"] = torch.cat([
        annotation["category"],
        torch.tensor([category], dtype=torch.int64)
    ])
    annotation["bbox"] = torch.cat([
        annotation["bbox"],
        torch.tensor([bbox], dtype=torch.float32)
    ])
    annotation["visibility"] = torch.cat([
        annotation["visibility"],
        torch.tensor([visibility], dtype=torch.float32)
    ])
    return annotation


def collate_fn(batch):
    images, annotations, metas = zip(*batch)    # (B, T, ...)
    _B = len(batch)
    _T = len(images[0])
    images_list = [clip[_] for clip in images for _ in range(len(clip))]
    size_divisibility = metas[0][0]["size_divisibility"]
    nested_tensor = nested_tensor_from_tensor_list(images_list, size_divisibility=size_divisibility)
    # Reshape the nested tensor:
    nested_tensor.tensors = einops.rearrange(
        nested_tensor.tensors, "(b t) c h w -> b t c h w", b=_B, t=_T
    )
    nested_tensor.mask = einops.rearrange(
        nested_tensor.mask, "(b t) h w -> b t h w", b=_B, t=_T
    )
    # Above is prepared for DETR.
    # Below is prepared for MOTIP, pre-padding the annotations:
    max_N = max(annotation[0]["trajectory_id_labels"].shape[-1] for annotation in annotations)
    # Padding the ID annotations:
    for b in range(len(annotations)):
        for t in range(len(annotations[b])):
            _G, _, _N = annotations[b][t]["trajectory_id_labels"].shape
            if _N < max_N:
                annotations[b][t]["trajectory_id_labels"] = torch.cat([
                    annotations[b][t]["trajectory_id_labels"],
                    - torch.ones((_G, 1, max_N - _N), dtype=torch.int64)
                ], dim=-1)
                annotations[b][t]["trajectory_id_masks"] = torch.cat([
                    annotations[b][t]["trajectory_id_masks"],
                    torch.ones((_G, 1, max_N - _N), dtype=torch.bool)
                ], dim=-1)
                annotations[b][t]["trajectory_ann_idxs"] = torch.cat([
                    annotations[b][t]["trajectory_ann_idxs"],
                    - torch.ones((_G, 1, max_N - _N), dtype=torch.int64)
                ], dim=-1)
                annotations[b][t]["trajectory_times"] = torch.cat([
                    annotations[b][t]["trajectory_times"],
                    t * torch.ones((_G, 1, max_N - _N), dtype=torch.int64)
                ], dim=-1)
                annotations[b][t]["unknown_id_labels"] = torch.cat([
                    annotations[b][t]["unknown_id_labels"],
                    - torch.ones((_G, 1, max_N - _N), dtype=torch.int64)
                ], dim=-1)
                annotations[b][t]["unknown_real_id_labels"] = torch.cat([
                    annotations[b][t]["unknown_real_id_labels"],
                    - torch.ones((_G, 1, max_N - _N), dtype=torch.int64)
                ], dim=-1)
                annotations[b][t]["unknown_id_masks"] = torch.cat([
                    annotations[b][t]["unknown_id_masks"],
                    torch.ones((_G, 1, max_N - _N), dtype=torch.bool)
                ], dim=-1)
                annotations[b][t]["unknown_ann_idxs"] = torch.cat([
                    annotations[b][t]["unknown_ann_idxs"],
                    - torch.ones((_G, 1, max_N - _N), dtype=torch.int64)
                ], dim=-1)
                annotations[b][t]["unknown_times"] = torch.cat([
                    annotations[b][t]["unknown_times"],
                    t * torch.ones((_G, 1, max_N - _N), dtype=torch.int64)
                ], dim=-1)
            pass
    return {
        "images": nested_tensor,
        "annotations": annotations,
        "metas": metas,
    }


def generate_scales(base_size: int, base_size_repeat: int, ratio_range=(0.75, 1.25), multiple: int = 32):
    """DEIM `generate_scales` (engine/data/dataloader.py): sizes spanning ~75%..125% of the base, multiples of 32,
    with the base size repeated `base_size_repeat` times (the 'bias toward base' of MOTIP2 App. A)."""
    lo = int(base_size * ratio_range[0] / multiple) * multiple
    hi = int(base_size * ratio_range[1] / multiple) * multiple
    scale_repeat = (base_size - lo) // multiple
    scales = [lo + i * multiple for i in range(scale_repeat)]
    scales += [base_size] * base_size_repeat
    scales += [hi - i * multiple for i in range(scale_repeat)]
    return scales


@torch.no_grad()
def batch_multiscale(images, scale_factor: float, multiple: int = 32):
    """Resize a (B, T, C, H, W) NestedTensor batch to ONE common size, shared by every frame of every clip.

    MOTIP2 App. A: "each clip is resized to one of several resolutions ... all frames in a clip share the sampled
    scale". DEIM has no padding-mask support, so per-clip valid regions are interpolated to a common size
    (derived from the first clip, times `scale_factor`, rounded to a multiple of `multiple`); the returned mask is
    all False (no padding). Boxes are normalised, so they need no change.
    """
    from utils.nested_tensor import NestedTensor
    x, m = images.tensors, images.mask            # (B,T,C,H,W), (B,T,H,W) (True = padding)
    B, T = x.shape[:2]
    valid_h = (~m[:, 0]).any(dim=-1).sum(-1)        # (B,)
    valid_w = (~m[:, 0]).any(dim=-2).sum(-1)
    th = max(multiple, int(round(valid_h[0].item() * scale_factor / multiple)) * multiple)
    tw = max(multiple, int(round(valid_w[0].item() * scale_factor / multiple)) * multiple)
    out = x.new_zeros((B, T, x.shape[2], th, tw))
    for b in range(B):
        h, w = int(valid_h[b]), int(valid_w[b])
        frames = x[b, :, :, :h, :w].float()
        out[b] = F.interpolate(frames, size=(th, tw), mode="bilinear", align_corners=False).to(x.dtype)
    return NestedTensor(tensors=out, mask=torch.zeros((B, T, th, tw), dtype=torch.bool, device=x.device))
