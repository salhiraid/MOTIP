# Gate 5 (unit, CPU): Spatial ID Switches. PLAN.md §6 Gate 5.
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.id_switch import apply_id_switch, pairwise_giou, spatial_permutation, SwitchStats
from data.transforms import GenerateIDLabels, TurnIntoTrajectoryAndUnknown


def scene(n=10, T=6, G=3, overlapping=True, seed=0):
    """n objects, static boxes. overlapping=True -> two clusters of strongly overlapping boxes; False -> far apart."""
    g = torch.Generator().manual_seed(seed)
    if overlapping:
        centers = torch.cat([torch.tensor([[0.3, 0.5]]) + 0.02 * torch.randn(n // 2, 2, generator=g),
                             torch.tensor([[0.75, 0.5]]) + 0.02 * torch.randn(n - n // 2, 2, generator=g)])
        wh = torch.full((n, 2), 0.2)
    else:
        centers = torch.stack([torch.tensor([0.05 + 0.1 * (i % 10), 0.1 + 0.4 * (i // 10)]) for i in range(n)])
        wh = torch.full((n, 2), 0.04)
    boxes = torch.cat([centers, wh], dim=-1)
    ann = torch.arange(n)[None, None, :].repeat(G, T, 1)
    masks = torch.zeros(G, T, n, dtype=torch.bool)
    return ann, masks, [boxes.clone() for _ in range(T)]


def test_giou_matrix_diagonal_one_and_symmetric():
    b = torch.rand(6, 4) * 0.3 + 0.2
    G = pairwise_giou(b)
    assert torch.allclose(G.diag(), torch.ones(6)) and torch.allclose(G, G.T, atol=1e-6)
    assert (G <= 1 + 1e-6).all() and (G >= -1 - 1e-6).all()


@pytest.mark.parametrize("sigma", [0.0, 0.5, 5.0, 1e6])
def test_permutation_is_valid(sigma):
    torch.manual_seed(0)
    for n in (2, 3, 7, 20):
        perm = spatial_permutation(pairwise_giou(torch.rand(n, 4) * 0.3 + 0.2), sigma)
        assert sorted(perm.tolist()) == list(range(n))


def test_sigma_zero_is_identity_when_boxes_are_distinct():
    # no noise: the diagonal (GIoU = 1) always wins -> nobody swaps
    perm = spatial_permutation(pairwise_giou(torch.rand(8, 4) * 0.3 + 0.2), 0.0)
    assert perm.tolist() == list(range(8))


def test_apply_is_a_permutation_within_each_frame_and_none_is_identity():
    torch.manual_seed(0)
    ann, masks, boxes = scene()
    a2, m2 = apply_id_switch(ann, masks, boxes, mode="none")
    assert torch.equal(a2, ann) and torch.equal(m2, masks)
    for mode in ("uniform", "spatial"):
        a2, m2 = apply_id_switch(ann, masks, boxes, mode=mode, sigma=0.5)
        assert torch.equal(a2.sort(-1).values, ann.sort(-1).values), mode      # same set of objects per (g, t)
        assert torch.equal(m2, masks)                                          # all present -> masks untouched
        assert not torch.equal(a2, ann)                                        # something did switch


def swap_rate(mode, sigma, draws, overlapping=True, **kw):
    torch.manual_seed(1)
    stats = SwitchStats()
    ann, masks, boxes = scene(overlapping=overlapping, G=6, T=8)
    for _ in range(draws):
        apply_id_switch(ann, masks, boxes, mode=mode, sigma=sigma, stats=stats, **kw)
    return stats.summary()


def test_sigma_infinity_recovers_uniform_swap_rate():
    uni = swap_rate("uniform", 0.5, 120)             # 120 calls x 48 (g,t) frames ~ 5.8k draws
    big = swap_rate("spatial", 1e6, 120)
    assert abs(uni["Swap%"] - big["Swap%"]) < 2.5, (uni, big)
    assert abs(uni["meanGIoU"] - big["meanGIoU"]) < 0.06, (uni, big)


def test_spatial_swaps_far_less_than_uniform_and_prefers_overlap():
    uni = swap_rate("uniform", 0.5, 60)
    sp = swap_rate("spatial", 0.5, 60)
    assert sp["Swap%"] < 0.6 * uni["Swap%"], (sp, uni)
    # swapped pairs of the spatial variant are mostly overlapping ones, uniform's are mostly distant (Tab. 8 trend)
    assert sp["LR%"] < uni["LR%"] and sp["meanGIoU"] > uni["meanGIoU"]


def test_non_overlapping_pairs_rarely_swap_at_sigma_half():
    sp = swap_rate("spatial", 0.5, 60, overlapping=False)
    assert sp["Swap%"] < 2.0, sp                     # analytic pair probability ~1e-3
    uni = swap_rate("uniform", 0.5, 20, overlapping=False)
    assert uni["Swap%"] > 20.0


def test_two_close_boxes_swap_with_substantial_probability():
    torch.manual_seed(0)
    boxes = torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.51, 0.5, 0.2, 0.2]])
    G = pairwise_giou(boxes)
    n_swap = sum(int(spatial_permutation(G, 0.5)[0] == 1) for _ in range(4000))
    assert 0.2 < n_swap / 4000 < 0.5                 # analytic ~0.35 for GIoU ~0.9


def test_skip_absent_never_touches_masked_slots_but_neg_one_may():
    torch.manual_seed(0)
    ann, masks, boxes = scene()
    masks[:, :, 3] = True                            # identity slot 3 is absent in every frame
    for _ in range(30):
        a2, m2 = apply_id_switch(ann, masks, boxes, mode="spatial", sigma=3.0, absent="skip")
        assert torch.equal(a2[:, :, 3], ann[:, :, 3]) and m2[:, :, 3].all()
    moved = False
    for _ in range(60):
        a2, m2 = apply_id_switch(ann, masks, boxes, mode="spatial", sigma=3.0, absent="neg_one", subset_prob=0.9)
        moved |= not torch.equal(m2[:, :, 3], masks[:, :, 3])
    assert moved


def test_last_frame_reference_applies_one_permutation_to_the_whole_history():
    torch.manual_seed(0)
    ann, masks, boxes = scene(G=1, T=6)
    for _ in range(20):
        a2, _ = apply_id_switch(ann, masks, boxes, mode="spatial", sigma=3.0, reference="last_frame", subset_prob=0.8)
        for t in range(1, 6):
            assert torch.equal(a2[0, t], a2[0, 0])    # same permutation in every frame (static scene)


def _legacy_uniform(trajectory_ann_idxs, trajectory_id_masks, prob):
    """MOTIP's original code from data/transforms.py (base commit e23387c), verbatim logic."""
    import einops
    G, T, N = trajectory_ann_idxs.shape
    a = einops.rearrange(trajectory_ann_idxs, "G T N -> (G T) N").clone()
    m = einops.rearrange(trajectory_id_masks, "G T N -> (G T) N").clone()
    for g_t in range(G * T):
        switch_p = torch.ones((N,)) * prob
        switch_map = torch.bernoulli(switch_p)
        switch_idxs = torch.nonzero(switch_map)[:, 0]
        if len(switch_idxs) > 1:
            shuffled_switch_idxs = switch_idxs[torch.randperm(len(switch_idxs))]
            a[g_t, switch_idxs] = a[g_t, shuffled_switch_idxs]
            m[g_t, switch_idxs] = m[g_t, shuffled_switch_idxs]
    return einops.rearrange(a, "(G T) N -> G T N", G=G, T=T), einops.rearrange(m, "(G T) N -> G T N", G=G, T=T)


def test_uniform_mode_is_bit_identical_to_original_motip_augmentation():
    ann, masks, boxes = scene()
    masks[0, 2, 1] = True                            # include a masked slot: legacy permutes masks too
    torch.manual_seed(123)
    la, lm = _legacy_uniform(ann, masks, 0.5)
    torch.manual_seed(123)
    na, nm = apply_id_switch(ann, masks, boxes, mode="uniform", subset_prob=0.5)
    assert torch.equal(la, na) and torch.equal(lm, nm)


def _annotations(T=6, n=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    anns = []
    for t in range(T):
        anns.append({"id": torch.arange(1, n + 1), "category": torch.zeros(n, dtype=torch.int64),
                     "bbox": torch.cat([torch.rand(n, 2, generator=g) * 0.6 + 0.2, torch.full((n, 2), 0.1)], -1),
                     "visibility": torch.ones(n)})
    return anns


def test_transform_integration_modes_and_legacy_default():
    images = [torch.zeros(3, 8, 8) for _ in range(6)]
    for kw, expected in [(dict(), "uniform"), (dict(aug_trajectory_switch_prob=0.0), "none")]:
        t = TurnIntoTrajectoryAndUnknown(num_id_vocabulary=20, aug_trajectory_occlusion_prob=0.5,
                                         **({"aug_trajectory_switch_prob": 0.5} | kw))
        assert t.id_switch == expected
    stats = SwitchStats()
    t = TurnIntoTrajectoryAndUnknown(num_id_vocabulary=20, aug_trajectory_occlusion_prob=0.5,
                                     aug_trajectory_switch_prob=0.5, id_switch="spatial", id_switch_sigma=0.5,
                                     id_switch_reference="per_frame", id_switch_subset_prob=0.5, stats=stats)
    anns = _annotations()
    _, anns, _ = GenerateIDLabels(num_id_vocabulary=20, aug_num_groups=6, num_training_ids=20)(images, anns, [{}] * 6)
    _, anns, _ = t(images, anns, [{}] * 6)
    a = anns[0]
    assert a["trajectory_ann_idxs"].shape == a["unknown_ann_idxs"].shape == (6, 1, 8)
    assert stats.n_obj > 0
