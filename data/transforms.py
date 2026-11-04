# Copyright (c) Ruopeng Gao. All Rights Reserved.

import copy
import math
import torch
import einops
import random
from torchvision.transforms import v2
import torchvision.transforms as T
from math import floor
from PIL import Image

from utils.box_ops import box_xywh_to_xyxy, box_xyxy_to_cxcywh
from .util import is_legal
from .id_switch import apply_id_switch, SwitchStats


def _tv_params(transform, flat_inputs):
    """torchvision >= 0.22 renamed `_get_params` -> `make_params` and `_transform` -> `transform`."""
    fn = getattr(transform, "make_params", None) or transform._get_params
    return fn(flat_inputs)


def _tv_apply(transform, inpt, params):
    fn = getattr(transform, "transform", None) or transform._transform
    return fn(inpt, params)


class MultiIdentity:
    def __init__(self):
        return

    def __call__(self, images, annotations, metas):
        return images, annotations, metas


class MultiCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images, annotations, metas):
        for transform in self.transforms:
            images, annotations, metas = transform(images, annotations, metas)
        return images, annotations, metas


# ---- per-detection keypoints (models/deim/keypoints.py) ------------------------------------------------------------
# An annotation may carry `keypoints` (N, K, 2) in the current image's pixels (normalised by MultiNormalize*) and
# `keypoints_vis` (N, K) = the GT visibility value (0 = not labelled). Every geometric transform below keeps them in step
# with the boxes; points that leave the image get visibility 0 (unlabelled).
def _kp_fields(annotation) -> list:
    return [f for f in ("keypoints", "keypoints_vis") if f in annotation]


def _kp_affine(annotation, scale=(1.0, 1.0), shift=(0.0, 0.0)):
    if "keypoints" in annotation:
        annotation["keypoints"] = annotation["keypoints"] * torch.as_tensor(scale, dtype=torch.float32) \
            + torch.as_tensor(shift, dtype=torch.float32)


def _kp_mask_outside(annotation, width, height):
    if "keypoints" in annotation:
        k = annotation["keypoints"]
        out = (k[..., 0] < 0) | (k[..., 0] > width) | (k[..., 1] < 0) | (k[..., 1] > height)
        annotation["keypoints_vis"] = annotation["keypoints_vis"].masked_fill(out, 0)


def flip_pair_permutation(num_keypoints: int, flip_pairs) -> list:
    """flip_pairs: [[i, j], ...] or a flat list [i, j, i, j, ...] of left/right keypoint indices."""
    if flip_pairs and not isinstance(flip_pairs[0], (list, tuple)):
        flip_pairs = [flip_pairs[i:i + 2] for i in range(0, len(flip_pairs), 2)]
    perm = list(range(num_keypoints))
    for i, j in flip_pairs or []:
        perm[i], perm[j] = j, i
    return perm


class MultiSimulate:
    """
    Simulate a video clip from a sequence of images.
    """
    def __init__(self, max_shift_ratio: float, overflow_bbox: bool):
        self.max_shift_ratio = max_shift_ratio
        self.overflow_bbox = overflow_bbox
        return

    def __call__(self, images, annotations, metas):
        if metas[0]["is_static"] is False:
            return images, annotations, metas
        else:
            # Currently, we simulate a video clip by shifting the images.
            # However, as discussed in MOTIP Appendix C.2, we need more advanced methods to simulate a video clip.

            # Calculate the shift meta infos:
            w, h = images[0].size
            max_x_shift = math.ceil(self.max_shift_ratio * w)
            max_y_shift = math.ceil(self.max_shift_ratio * h)
            x_shift = random.randint(-max_x_shift, max_x_shift)
            y_shift = random.randint(-max_y_shift, max_y_shift)
            # Prepare for the shifted sequence:
            shifted_images, shifted_annotations = [], []
            shifted_images.append(copy.deepcopy(images[0]))
            shifted_annotations.append(copy.deepcopy(annotations[0]))
            # Shifting for the rest images:
            for _idx in range(1, len(images)):
                x_min, x_max = max(0, x_shift), min(w, w + x_shift)
                y_min, y_max = max(0, y_shift), min(h, h + y_shift)
                _image = copy.deepcopy(shifted_images[_idx - 1])
                _ann = copy.deepcopy(shifted_annotations[_idx - 1])
                # Crop:
                _i, _j, _h, _w = y_min, x_min, y_max - y_min, x_max - x_min
                _ann["bbox"] = _ann["bbox"] - torch.tensor([_j, _i, _j, _i])
                _kp_affine(_ann, shift=(-_j, -_i))
                _kp_mask_outside(_ann, _w, _h)
                _bbox = _ann["bbox"].clone()
                _max_wh = torch.tensor([_w, _h])
                _bbox = torch.min(_bbox.reshape(-1, 2, 2), _max_wh)
                _bbox = _bbox.clamp(min=0)
                _legal_idxs = torch.all(_bbox[:, 1, :] > _bbox[:, 0, :], dim=1)
                # Reshape to the original format:
                _bbox = _bbox.reshape(-1, 4)
                _need_to_select_fields = ["bbox", "category", "id", "visibility"] + _kp_fields(_ann)
                if self.overflow_bbox is False:
                    _ann["bbox"] = _bbox
                for _field in _need_to_select_fields:
                    _ann[_field] = _ann[_field][_legal_idxs]
                _ann["is_legal"] = is_legal(_ann)
                _image = v2.functional.crop(_image, _i, _j, _h, _w)
                # Resize:
                _h_ratio = h / _h
                _w_ratio = w / _w
                _bbox_ratio = torch.tensor([_w_ratio, _h_ratio] * 2)
                _ann["bbox"] = _ann["bbox"] * _bbox_ratio
                _kp_affine(_ann, scale=(_w_ratio, _h_ratio))
                _image = v2.functional.resize(_image, [h, w])
                # Put into the shifted sequence:
                shifted_images.append(_image)
                shifted_annotations.append(_ann)
            # Check if the shifted sequence is legal:
            _is_legals = torch.tensor([_ann["is_legal"] for _ann in shifted_annotations])
            if not _is_legals.all().item():
                return images, annotations, metas
            else:
                if random.random() < 0.5:
                    return shifted_images, shifted_annotations, metas
                else:
                    # Inverse the sequence:
                    shifted_images = shifted_images[::-1]
                    shifted_annotations = shifted_annotations[::-1]
                    metas = metas[::-1]
                    # We need to fix the "is_begin" field:
                    _meta_begins = [meta["is_begin"] for meta in metas]
                    _meta_begins = _meta_begins[::-1]
                    for _ in range(len(metas)):
                        metas[_]["is_begin"] = _meta_begins[_]
                    return shifted_images, shifted_annotations, metas


class MultiStack:
    """
    Stack a sequence of images into a single tensor, (T, C, H, W).
    The result tensor is more suitable for multi-image processing.
    """
    def __init__(self):
        return

    def __call__(self, images, annotations, metas):
        if isinstance(images, list):
            if isinstance(images[0], torch.Tensor):
                images = torch.stack(images, dim=0)
        return images, annotations, metas


class MultiBoxXYWHtoXYXY:
    def __init__(self):
        return

    def __call__(self, images, annotations, metas):
        for _ in range(len(annotations)):
            annotations[_]["bbox"] = box_xywh_to_xyxy(annotations[_]["bbox"])
        return images, annotations, metas


class MultiBoxXYXYtoCXCYWH:
    def __init__(self):
        return

    def __call__(self, images, annotations, metas):
        for _ in range(len(annotations)):
            annotations[_]["bbox"] = box_xyxy_to_cxcywh(annotations[_]["bbox"])
        return images, annotations, metas


class MultiRandomHorizontalFlip:
    def __init__(self, p=0.5, keypoint_flip_pairs=None):
        self.p = p
        self.keypoint_flip_pairs = keypoint_flip_pairs

    def __call__(self, images, annotations, metas):
        # Here, the boxes in annotations are in the format of (x1, y1, x2, y2).
        if torch.rand(1).item() < self.p:
            if isinstance(images, torch.Tensor):
                images = v2.functional.horizontal_flip_image(images)
            elif isinstance(images, list):
                assert isinstance(images[0], Image.Image)
                images = [v2.functional.hflip(_) for _ in images]
            else:
                raise NotImplementedError(f"The input image type {type(images)} is not supported.")
            h, w = get_image_hw(images)
            for annotation in annotations:
                annotation["bbox"] = (
                    annotation["bbox"][:, [2, 1, 0, 3]]
                    * torch.as_tensor([-1, 1, -1, 1])
                    + torch.as_tensor([w, 0, w, 0])
                )
                if "keypoints" in annotation:
                    if not self.keypoint_flip_pairs:
                        raise ValueError("Horizontal flip with keypoints needs KEYPOINT_FLIP_PAIRS "
                                         "(otherwise left/right keypoints would be mislabelled).")
                    perm = flip_pair_permutation(annotation["keypoints"].shape[1], self.keypoint_flip_pairs)
                    annotation["keypoints"] = annotation["keypoints"][:, perm] * torch.as_tensor([-1.0, 1.0]) \
                        + torch.as_tensor([float(w), 0.0])
                    annotation["keypoints_vis"] = annotation["keypoints_vis"][:, perm]
        return images, annotations, metas


class MultiRandomSelect:
    def __init__(self, transform1, transform2, p: float = 0.5):
        self.transform1 = transform1
        self.transform2 = transform2
        self.p = p

    def __call__(self, images, annotations, metas):
        if torch.rand(1).item() < self.p:
            return self.transform1(images, annotations, metas)
        else:
            return self.transform2(images, annotations, metas)


class MultiRandomResize:
    def __init__(self, sizes: list, max_size: int | None = None):
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, images, annotations, metas):
        new_shorter = random.choice(self.sizes)     # choose the shorter edge length for images

        def get_new_hw(_curr_hw: list, _new_shorter) -> (int, int):
            _curr_h, _curr_w = _curr_hw
            if self.max_size is not None:       # need to restrict the longer side length
                _min_hw, _max_hw = float(min(_curr_h, _curr_w)), float(max(_curr_h, _curr_w))
                if _max_hw / _min_hw * _new_shorter > self.max_size:  # need to restrict the resize size
                    _new_shorter = int(floor(self.max_size * _min_hw / _max_hw))
            # Calculate the new height and width:
            if _curr_w < _curr_h:
                _new_w = _new_shorter
                _new_h = int(round(_new_shorter * _curr_h / _curr_w))
            else:
                _new_h = _new_shorter
                _new_w = int(round(_new_shorter * _curr_w / _curr_h))
            return _new_h, _new_w

        new_hw = get_new_hw(get_image_hw(images), _new_shorter=new_shorter)    # new yx
        scale_ratio_x = new_hw[1] / get_image_hw(images)[1]
        scale_ratio_y = new_hw[0] / get_image_hw(images)[0]
        # Resize images:
        if isinstance(images, torch.Tensor):
            images = v2.functional.resize(images, new_hw)
        elif isinstance(images, list):
            assert isinstance(images[0], Image.Image)
            images = [v2.functional.resize(_, new_hw) for _ in images]
        else:
            raise NotImplementedError(f"The input image type {type(images)} is not supported.")
        # Resize annotations:
        for annotation in annotations:
            annotation["bbox"] = annotation["bbox"] * torch.as_tensor([scale_ratio_x, scale_ratio_y] * 2)
            _kp_affine(annotation, scale=(scale_ratio_x, scale_ratio_y))
        return images, annotations, metas


class MultiRandomCrop:
    def __init__(self, min_size: int, max_size: int, overflow_bbox: bool):
        self.min_size = min_size
        self.max_size = max_size
        self.overflow_bbox = overflow_bbox

    def __call__(self, images, annotations, metas):
        # Calculate the crop box:
        curr_h, curr_w = get_image_hw(images)
        crop_h = random.randint(self.min_size, min(self.max_size, curr_h))
        crop_w = random.randint(self.min_size, min(self.max_size, curr_w))
        crop_ijhw = T.RandomCrop.get_params(images[0], (crop_h, crop_w))

        # Crop the cropped annotations:
        _annotations = copy.deepcopy(annotations)
        _i, _j, _h, _w = crop_ijhw
        for _annotation in _annotations:
            _annotation["bbox"] = _annotation["bbox"] - torch.tensor([_j, _i, _j, _i])  # (x1,y1,x2,y2) - (j,i,j,i)
            _kp_affine(_annotation, shift=(-_j, -_i))
            _kp_mask_outside(_annotation, _w, _h)
            _bbox = _annotation["bbox"].clone()
            _max_wh = torch.tensor([_w, _h])
            # If the crop box is out of the image, we need to adjust the bbox:
            _bbox = torch.min(_bbox.reshape(-1, 2, 2), _max_wh)
            _bbox = _bbox.clamp(min=0)
            # We need to find the legal bbox:
            # _legal_idxs = torch.all(
            #     torch.tensor(_bbox[:, 1, :] > _bbox[:, 0, :]), dim=1
            # )
            _legal_idxs = torch.all(
                _bbox[:, 1, :] > _bbox[:, 0, :], dim=1
            )
            # Reshape to the original format:
            _bbox = _bbox.reshape(-1, 4)
            _need_to_select_fields = ["bbox", "category", "id", "visibility"] + _kp_fields(_annotation)
            if self.overflow_bbox is False:
                _annotation["bbox"] = _bbox
            for _field in _need_to_select_fields:
                _annotation[_field] = _annotation[_field][_legal_idxs]
            _annotation["is_legal"] = is_legal(_annotation)

        # Check all annotations' legality:
        _is_legals = torch.tensor([_annotation["is_legal"] for _annotation in _annotations])
        # If all annotations are illegal, we need to return the original images and annotations:
        if not _is_legals.all().item():
            return images, annotations, metas
        else:
            if isinstance(images, torch.Tensor):
                images = v2.functional.crop(images, _i, _j, _h, _w)
            elif isinstance(images, list):
                assert isinstance(images[0], Image.Image)
                images = [v2.functional.crop(_, _i, _j, _h, _w) for _ in images]
            else:
                raise NotImplementedError(f"The input image type {type(images)} is not supported.")
            annotations = _annotations
            return images, annotations, metas


class MultiRandomZoomOut:
    """MOTIP2 Paper App. A: random zoom-out p=0.5, scale in [1.0, 2.0] (DEIM RandomZoomOut semantics).
    The same canvas/offset is used for every frame of the clip. Boxes are (x1, y1, x2, y2) in pixels."""
    def __init__(self, p: float = 0.5, scale_range=(1.0, 2.0), fill: int = 0):
        self.p, self.scale_range, self.fill = p, scale_range, fill

    def __call__(self, images, annotations, metas):
        if random.random() >= self.p:
            return images, annotations, metas
        assert isinstance(images, list) and isinstance(images[0], Image.Image)
        w, h = images[0].size
        r = random.uniform(*self.scale_range)
        cw, ch = int(round(w * r)), int(round(h * r))
        left, top = random.randint(0, cw - w), random.randint(0, ch - h)
        new_images = []
        for im in images:
            canvas = Image.new(im.mode, (cw, ch), (self.fill,) * len(im.getbands()) if len(im.getbands()) > 1 else self.fill)
            canvas.paste(im, (left, top))
            new_images.append(canvas)
        shift = torch.as_tensor([left, top, left, top], dtype=torch.float32)
        for annotation in annotations:
            annotation["bbox"] = annotation["bbox"] + shift
            _kp_affine(annotation, shift=(left, top))
        return new_images, annotations, metas


class MultiRandomIoUCrop:
    """MOTIP2 Paper App. A: IoU-aware random crop p=0.8, scale [0.3, 1.0], aspect ratio [1.5, 2.0].
    Sampling follows torchvision.RandomIoUCrop on the first frame; the crop is then shared by the whole clip and
    accepted only if every frame keeps a legal annotation. Boxes are (x1, y1, x2, y2) in pixels."""
    def __init__(self, p: float = 0.8, min_scale: float = 0.3, max_scale: float = 1.0,
                 min_aspect_ratio: float = 1.5, max_aspect_ratio: float = 2.0,
                 sampler_options=(0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0), trials: int = 40):
        self.p, self.min_scale, self.max_scale = p, min_scale, max_scale
        self.min_ar, self.max_ar = min_aspect_ratio, max_aspect_ratio
        self.options, self.trials = list(sampler_options), trials

    @staticmethod
    def _crop_annotation(annotation, left, top, cw, ch):
        a = copy.copy(annotation)
        boxes = annotation["bbox"]
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        keep = (centers[:, 0] > left) & (centers[:, 0] < left + cw) & (centers[:, 1] > top) & (centers[:, 1] < top + ch)
        boxes = boxes[keep] - torch.as_tensor([left, top, left, top], dtype=torch.float32)
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, cw)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, ch)
        ok = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        keep_idx = torch.nonzero(keep)[:, 0][ok]
        a["bbox"] = boxes[ok]
        for f in ["category", "id", "visibility"] + _kp_fields(annotation):
            a[f] = annotation[f][keep_idx]
        if "keypoints" in a:
            _kp_affine(a, shift=(-left, -top))
            _kp_mask_outside(a, cw, ch)
        a["is_legal"] = is_legal(a)
        return a

    def __call__(self, images, annotations, metas):
        if random.random() >= self.p:
            return images, annotations, metas
        assert isinstance(images, list) and isinstance(images[0], Image.Image)
        w, h = images[0].size
        from torchvision.ops import box_iou
        boxes0 = annotations[0]["bbox"]
        while True:
            min_jaccard = self.options[random.randrange(len(self.options))]
            if min_jaccard >= 1.0:
                return images, annotations, metas
            for _ in range(self.trials):
                r = self.min_scale + (self.max_scale - self.min_scale) * torch.rand(2)
                cw, ch = int(w * r[0]), int(h * r[1])
                if cw < 1 or ch < 1 or not (self.min_ar <= cw / ch <= self.max_ar):
                    continue
                left, top = int(torch.randint(0, w - cw + 1, (1,))), int(torch.randint(0, h - ch + 1, (1,)))
                crop = torch.as_tensor([[left, top, left + cw, top + ch]], dtype=torch.float32)
                if len(boxes0) == 0 or box_iou(boxes0, crop).max() < min_jaccard:
                    continue
                cropped = [self._crop_annotation(a, left, top, cw, ch) for a in annotations]
                if not all(a["is_legal"] for a in cropped):
                    continue
                images = [im.crop((left, top, left + cw, top + ch)) for im in images]
                return images, cropped, metas


class MultiColorJitter:
    def __init__(
            self,
            brightness: float = 0.0,
            contrast: float = 0.0,
            saturation: float = 0.0,
            hue: float = 0.0
    ):
        self.color_jitter = v2.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )

    def __call__(self, images, annotations, metas):
        if isinstance(images, torch.Tensor):
            images = self.color_jitter(images)
        elif isinstance(images, list):
            assert isinstance(images[0], Image.Image)
            params = _tv_params(self.color_jitter, [images[0]])
            images = [_tv_apply(self.color_jitter, _, params) for _ in images]
        else:
            raise NotImplementedError(f"The input image type {type(images)} is not supported.")
        return images, annotations, metas


class MultiRandomPhotometricDistort:
    def __init__(self):
        self.ramdom_photometric_distort = v2.RandomPhotometricDistort()

    def __call__(self, images, annotations, metas):
        _params = _tv_params(self.ramdom_photometric_distort, [images[0]])
        images = [_tv_apply(self.ramdom_photometric_distort, _, _params) for _ in images]
        return images, annotations, metas


class MultiToTensor:
    def __init__(self):
        return

    def __call__(self, images, annotations, metas):
        if isinstance(images, list):
            assert isinstance(images[0], Image.Image)
            images = [v2.functional.to_image(_) for _ in images]
        return images, annotations, metas


class MultiToDtype:
    def __init__(self, dtype: torch.dtype):
        self.dtype = dtype
        return

    def __call__(self, images, annotations, metas):
        if isinstance(images, torch.Tensor):
            images = v2.functional.to_dtype(images, dtype=torch.float32, scale=True)
        elif isinstance(images, list):
            assert isinstance(images[0], torch.Tensor)
            images = [v2.functional.to_dtype(_, dtype=torch.float32, scale=True) for _ in images]
        else:
            raise NotImplementedError(f"The input image type {type(images)} is not supported.")
        return images, annotations, metas


class MultiNormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, images, annotations, metas):
        # images = images.to(torch.float32).div(255)
        # images = v2.functional.normalize(images, mean=self.mean, std=self.std)
        h, w = images.shape[-2:]
        for annotation in annotations:
            annotation["bbox"] = annotation["bbox"] / torch.tensor([w, h, w, h])
            _kp_affine(annotation, scale=(1.0 / w, 1.0 / h))
        return images.contiguous(), annotations, metas


class MultiNormalizeBoundingBoxes:
    def __init__(self):
        return

    def __call__(self, images, annotations, metas):
        # Only normalize the bounding boxes,
        # the images will be normalized in the training loop (on cuda).
        h, w = images.shape[-2:]
        for annotation in annotations:
            annotation["bbox"] = annotation["bbox"] / torch.tensor([w, h, w, h])
            _kp_affine(annotation, scale=(1.0 / w, 1.0 / h))
        return images.contiguous(), annotations, metas


# For MOTIP only, biding the ID label:

class GenerateIDLabels:
    def __init__(self, num_id_vocabulary: int, aug_num_groups: int, num_training_ids: int):
        self.num_id_vocabulary = num_id_vocabulary
        self.aug_num_groups = aug_num_groups
        self.num_training_ids = num_training_ids

    def __call__(self, images, annotations, metas):
        _T = len(images)
        _G = self.aug_num_groups
        # Collect all IDs:
        ids_set = set()
        for annotation in annotations:
            ids_set.update(set(annotation["id"].tolist()))
        _N = len(ids_set)

        # ID anns consist of the following parts:
        # (1): a (_G, _T, _N) tensor, representing the ID labels of each object in each frame.
        # (2): a (_G, _T, _N) tensor, representing the corresponding index of each object in detection annotations.
        # (3): a (_G, _T, _N) tensor, representing the mask of ID labels in each frame.
        # (4): a (_G, _T, _N) tensor, representing the time index of each object.

        ids_list = list(ids_set)
        id_to_idx = {ids_list[_]: _ for _ in range(_N)}     # the idx in the final ID labels
        base_id_masks = torch.ones((_T, _N), dtype=torch.bool)
        base_ann_idxs = - torch.ones((_T, _N), dtype=torch.int64)
        # These "base" ID anns are used to generate the final ID anns, do not directly use them.
        for t in range(_T):
            annotation = annotations[t]
            for i in range(len(annotation["id"])):
                _id = annotation["id"][i].item()
                _ann_idx = i
                _n = id_to_idx[_id]
                # generate the corresponding ID ann:
                base_id_masks[t, _n] = False
                base_ann_idxs[t, _n] = _ann_idx

        # Generate the final ID anns
        # If the number of IDs is larger than `num_id_vocabulary`, we need to randomly select a subset of IDs.
        # Also, if the number of IDs is larger than `num_training_ids`, we need to randomly select a subset of IDs.
        if _N > self.num_id_vocabulary or _N > self.num_training_ids:
            _random_select_idxs = torch.randperm(_N)[:self.num_training_ids if _N > self.num_training_ids else self.num_id_vocabulary]
            base_id_masks = base_id_masks[:, _random_select_idxs]
            base_ann_idxs = base_ann_idxs[:, _random_select_idxs]
            _N = self.num_training_ids if _N > self.num_training_ids else self.num_id_vocabulary
            pass
        # Normal processing:
        id_labels = torch.zeros((_G, _T, _N), dtype=torch.int64)
        id_masks = torch.ones((_G, _T, _N), dtype=torch.bool)
        ann_idxs = - torch.ones((_G, _T, _N), dtype=torch.int64)
        for group in range(_G):
            _random_id_labels = torch.randperm(self.num_id_vocabulary)[:_N]
            _random_id_labels = _random_id_labels[None, ...].repeat(_T, 1)
            # _random_id_labels[base_id_masks] = -1
            id_labels[group] = _random_id_labels.clone()
            id_masks[group] = base_id_masks.clone()
            ann_idxs[group] = base_ann_idxs.clone()
        # Generate the time indexes:
        times = torch.arange(_T, dtype=torch.int64)[None, :, None].repeat(_G, 1, _N)
        # Check the shapes:
        assert id_labels.shape == id_masks.shape == ann_idxs.shape == times.shape

        # Split the ID anns into each frame:
        id_labels_list = torch.split(id_labels, split_size_or_sections=1, dim=1)    # each item is in (_G, 1, _N)
        id_masks_list = torch.split(id_masks, split_size_or_sections=1, dim=1)      # each item is in (_G, 1, _N)
        ann_idxs_list = torch.split(ann_idxs, split_size_or_sections=1, dim=1)      # each item is in (_G, 1, _N)
        times_list = torch.split(times, split_size_or_sections=1, dim=1)            # each item is in (_G, 1, _N)

        # Update the annotations (put the ID anns into the annotations):
        for t in range(_T):
            annotations[t]["id_labels"] = id_labels_list[t]
            annotations[t]["id_masks"] = id_masks_list[t]
            annotations[t]["ann_idxs"] = ann_idxs_list[t]
            annotations[t]["times"] = times_list[t]
        pass
        return images, annotations, metas


class TurnIntoTrajectoryAndUnknown:
    def __init__(
            self,
            num_id_vocabulary: int,
            aug_trajectory_occlusion_prob: float,
            aug_trajectory_switch_prob: float,
            # MOTIP2 (PLAN Phase 5). id_switch=None keeps MOTIP's behaviour: uniform iff aug_trajectory_switch_prob > 0.
            id_switch: str | None = None,               # none | uniform | spatial
            id_switch_sigma: float = 0.5,
            id_switch_reference: str = "per_frame",     # per_frame | last_frame
            id_switch_subset_prob: float | None = None,
            id_switch_absent: str = "skip",             # skip | neg_one
            stats=None,
    ):
        self.num_id_vocabulary = num_id_vocabulary
        self.aug_trajectory_occlusion_prob = aug_trajectory_occlusion_prob
        self.aug_trajectory_switch_prob = aug_trajectory_switch_prob
        if id_switch is None:
            id_switch = "uniform" if aug_trajectory_switch_prob > 0.0 else "none"
        self.id_switch = id_switch
        self.id_switch_sigma = id_switch_sigma
        self.id_switch_reference = id_switch_reference
        self.id_switch_subset_prob = aug_trajectory_switch_prob if id_switch_subset_prob is None else id_switch_subset_prob
        self.id_switch_absent = id_switch_absent
        self.stats = stats
        return

    def __call__(self, images, annotations, metas):
        id_labels = torch.cat([annotation["id_labels"] for annotation in annotations], dim=1)
        id_masks = torch.cat([annotation["id_masks"] for annotation in annotations], dim=1)
        ann_idxs = torch.cat([annotation["ann_idxs"] for annotation in annotations], dim=1)
        times = torch.cat([annotation["times"] for annotation in annotations], dim=1)
        _G, _T, _N = id_labels.shape
        # Del these fields from the annotations:
        for t in range(_T):
            del annotations[t]["id_labels"]
            del annotations[t]["id_masks"]
            del annotations[t]["ann_idxs"]
            del annotations[t]["times"]

        # Copy the ID anns to "trajectory_" and "unknown_":
        trajectory_id_labels = id_labels.clone()
        trajectory_id_masks = id_masks.clone()
        trajectory_ann_idxs = ann_idxs.clone()
        trajectory_times = times.clone()
        unknown_id_labels = id_labels.clone()
        unknown_real_id_labels = id_labels.clone()      # MOTIP2: identity labels before the newborn overwrite (C_i)
        unknown_id_masks = id_masks.clone()
        unknown_ann_idxs = ann_idxs.clone()
        unknown_times = times.clone()

        if self.aug_trajectory_occlusion_prob > 0.0:
            # Make trajectory occlusion:
            # 1. Turn the shape into (_G * _N, _T):
            trajectory_id_masks = einops.rearrange(trajectory_id_masks, "G T N -> (G N) T")
            unknown_id_masks = einops.rearrange(unknown_id_masks, "G T N -> (G N) T")
            # 2. Generate the occlusion mask:
            trajectory_occlusion_masks = torch.zeros_like(trajectory_id_masks, dtype=torch.bool)
            unknown_occlusion_masks = torch.zeros_like(unknown_id_masks, dtype=torch.bool)
            for i in range(_G * _N):
                if random.random() < self.aug_trajectory_occlusion_prob:
                    begin_idx = random.randint(0, _T - 1)
                    _max_T = _T - 1 - begin_idx
                    end_idx = begin_idx + math.ceil(_max_T * random.random())
                    trajectory_occlusion_masks[i, begin_idx:end_idx] = True
                    unknown_occlusion_masks[i, begin_idx:end_idx] = True
            # Currently, we do not check the legality of the occlusion mask.
            # However, we did it in the previous version.
            # 3. Apply the occlusion mask:
            trajectory_id_masks = trajectory_id_masks | trajectory_occlusion_masks
            unknown_id_masks = unknown_id_masks | unknown_occlusion_masks
            # 4. Turn the shape back:
            trajectory_id_masks = einops.rearrange(trajectory_id_masks, "(G N) T -> G T N", G=_G, N=_N)
            unknown_id_masks = einops.rearrange(unknown_id_masks, "(G N) T -> G T N", G=_G, N=_N)

        if self.id_switch != "none":
            # Make trajectory switch (MOTIP2 Sec. 3.2: `spatial` = GIoU-biased permutation, `uniform` = MOTIP's).
            # Switching the ID labels is the same as switching the ann_idxs and masks of the slots.
            _local_stats = SwitchStats()           # per-sample counts, aggregated in the training loop (Tab. 8 live)
            trajectory_ann_idxs, trajectory_id_masks = apply_id_switch(
                ann_idxs=trajectory_ann_idxs,
                id_masks=trajectory_id_masks,
                gt_boxes=[annotation["bbox"] for annotation in annotations],
                mode=self.id_switch,
                sigma=self.id_switch_sigma,
                reference=self.id_switch_reference,
                subset_prob=self.id_switch_subset_prob,
                absent=self.id_switch_absent,
                stats=_local_stats,
            )
            annotations[0]["switch_stats"] = {
                "n_obj": _local_stats.n_obj, "n_swapped": _local_stats.n_swapped,
                "n_long_range": _local_stats.n_long_range, "giou_sum": _local_stats.giou_sum,
            }
            if self.stats is not None:
                self.stats.n_obj += _local_stats.n_obj
                self.stats.n_swapped += _local_stats.n_swapped
                self.stats.n_long_range += _local_stats.n_long_range
                self.stats.giou_sum += _local_stats.giou_sum

        # Check all ID labels are legal:
        assert torch.all(trajectory_id_labels >= 0)
        assert torch.all(unknown_id_labels >= 0)

        # Add "newborn" ID label to unknown ID labels for supervision:
        # 1. Turn the shape into (_G * _N, _T):
        trajectory_id_labels = einops.rearrange(trajectory_id_labels, "G T N -> (G N) T")
        trajectory_id_masks = einops.rearrange(trajectory_id_masks, "G T N -> (G N) T")
        unknown_id_labels = einops.rearrange(unknown_id_labels, "G T N -> (G N) T")
        unknown_id_masks = einops.rearrange(unknown_id_masks, "G T N -> (G N) T")
        # 2. Calculate the already_born masks:
        already_born_masks = torch.cumsum(~trajectory_id_masks, dim=1)
        already_born_masks = already_born_masks > 0
        # 3. Generate the newborn ID labels:
        newborn_id_label_masks = ~ torch.cat(
            [
                torch.zeros((_G * _N, 1), dtype=torch.bool),
                already_born_masks[:, :-1]
            ],
            dim=-1
        )
        unknown_id_labels[newborn_id_label_masks] = self.num_id_vocabulary
        # 4. Turn the shape back:
        trajectory_id_labels = einops.rearrange(trajectory_id_labels, "(G N) T -> G T N", G=_G, N=_N)
        trajectory_id_masks = einops.rearrange(trajectory_id_masks, "(G N) T -> G T N", G=_G, N=_N)
        unknown_id_labels = einops.rearrange(unknown_id_labels, "(G N) T -> G T N", G=_G, N=_N)
        unknown_id_masks = einops.rearrange(unknown_id_masks, "(G N) T -> G T N", G=_G, N=_N)

        # Update the annotations:
        for t in range(_T):
            annotations[t]["trajectory_id_labels"] = trajectory_id_labels[:, t:t+1, :]
            annotations[t]["trajectory_id_masks"] = trajectory_id_masks[:, t:t+1, :]
            annotations[t]["trajectory_ann_idxs"] = trajectory_ann_idxs[:, t:t+1, :]
            annotations[t]["trajectory_times"] = trajectory_times[:, t:t+1, :]
            annotations[t]["unknown_id_labels"] = unknown_id_labels[:, t:t+1, :]
            annotations[t]["unknown_real_id_labels"] = unknown_real_id_labels[:, t:t+1, :]
            annotations[t]["unknown_id_masks"] = unknown_id_masks[:, t:t+1, :]
            annotations[t]["unknown_ann_idxs"] = unknown_ann_idxs[:, t:t+1, :]
            annotations[t]["unknown_times"] = unknown_times[:, t:t+1, :]

        return images, annotations, metas


def build_transforms(config: dict):
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]
    if config.get("AUG_PIPELINE", "legacy") == "motip2":
        return _build_motip2_transforms(config)
    return MultiCompose([
        MultiBoxXYWHtoXYXY(),
        MultiSimulate(max_shift_ratio=config["AUG_MAX_SHIFT_RATIO"], overflow_bbox=config["AUG_OVERFLOW_BBOX"]),
        MultiStack(),
        MultiRandomHorizontalFlip(p=0.5, keypoint_flip_pairs=config.get("KEYPOINT_FLIP_PAIRS")),
        MultiRandomSelect(
            MultiRandomResize(sizes=config["AUG_RESIZE_SCALES"], max_size=config["AUG_MAX_SIZE"]),
            MultiCompose([
                MultiRandomResize(sizes=config["AUG_RANDOM_RESIZE"]),
                MultiRandomCrop(
                    min_size=config["AUG_RANDOM_CROP_MIN"],
                    max_size=config["AUG_RANDOM_CROP_MAX"],
                    overflow_bbox=config["AUG_OVERFLOW_BBOX"]
                ),
                MultiRandomResize(sizes=config["AUG_RESIZE_SCALES"], max_size=config["AUG_MAX_SIZE"])
            ])
        ),
        MultiBoxXYXYtoCXCYWH(),
        MultiColorJitter(
            brightness=config["AUG_BRIGHTNESS"],
            contrast=config["AUG_CONTRAST"],
            saturation=config["AUG_SATURATION"],
            hue=config["AUG_HUE"],
        ) if not config["AUG_COLOR_JITTER_V2"] else MultiRandomPhotometricDistort(),
        MultiToTensor(),
        MultiStack(),
        # MultiToDtype(torch.float32),
        # MultiNormalize(mean=mean, std=std),
        MultiNormalizeBoundingBoxes(),
        # For MOTIP, biding ID labels:
        GenerateIDLabels(
            num_id_vocabulary=config["NUM_ID_VOCABULARY"],
            aug_num_groups=config["AUG_NUM_GROUPS"],
            num_training_ids=config.get("NUM_TRAINING_IDS", config["NUM_ID_VOCABULARY"]),
        ),
        TurnIntoTrajectoryAndUnknown(
            num_id_vocabulary=config["NUM_ID_VOCABULARY"],
            aug_trajectory_occlusion_prob=config["AUG_TRAJECTORY_OCCLUSION_PROB"],
            aug_trajectory_switch_prob=config["AUG_TRAJECTORY_SWITCH_PROB"],
        ),
    ])


def _flip_transforms(config: dict) -> list:
    """Horizontal flip; with keypoints it needs the left/right pairs, otherwise it is left out (a flip without the
    pairs would swap the meaning of left/right keypoints)."""
    if int(config.get("NUM_KEYPOINTS", 0) or 0) > 0 and not config.get("KEYPOINT_FLIP_PAIRS"):
        import warnings
        warnings.warn("NUM_KEYPOINTS > 0 and KEYPOINT_FLIP_PAIRS is not set: horizontal-flip augmentation is DISABLED")
        return []
    return [MultiRandomHorizontalFlip(p=0.5, keypoint_flip_pairs=config.get("KEYPOINT_FLIP_PAIRS"))]


def _build_motip2_transforms(config: dict):
    """MOTIP2 Paper App. A image pipeline (multi-scale is applied per batch, see data/util.py:batch_multiscale):
    flip p=0.5 -> zoom-out p=0.5 [1,2] -> IoU-crop p=0.8 [0.3,1] aspect [1.5,2] -> resize to base (shorter
    AUG_BASE_SIZE, longer <= AUG_MAX_SIZE) -> photometric p=0.5 (brightness .875-1.125, contrast/saturation .5-1.5,
    hue +-.05 = torchvision v2.RandomPhotometricDistort defaults)."""
    return MultiCompose([
        MultiBoxXYWHtoXYXY(),
        MultiSimulate(max_shift_ratio=config["AUG_MAX_SHIFT_RATIO"], overflow_bbox=config["AUG_OVERFLOW_BBOX"]),
        MultiStack(),
        *_flip_transforms(config),
        MultiRandomZoomOut(p=0.5, scale_range=(1.0, 2.0)),
        MultiRandomIoUCrop(p=0.8, min_scale=0.3, max_scale=1.0, min_aspect_ratio=1.5, max_aspect_ratio=2.0),
        MultiRandomResize(sizes=[config["AUG_BASE_SIZE"]], max_size=config["AUG_MAX_SIZE"]),
        MultiBoxXYXYtoCXCYWH(),
        MultiRandomPhotometricDistort(),
        MultiToTensor(),
        MultiStack(),
        MultiNormalizeBoundingBoxes(),
        GenerateIDLabels(
            num_id_vocabulary=config["NUM_ID_VOCABULARY"],
            aug_num_groups=config["AUG_NUM_GROUPS"],
            num_training_ids=config.get("NUM_TRAINING_IDS", config["NUM_ID_VOCABULARY"]),
        ),
        TurnIntoTrajectoryAndUnknown(
            num_id_vocabulary=config["NUM_ID_VOCABULARY"],
            aug_trajectory_occlusion_prob=config["AUG_TRAJECTORY_OCCLUSION_PROB"],
            aug_trajectory_switch_prob=config["AUG_TRAJECTORY_SWITCH_PROB"],
            id_switch=config["ID_SWITCH"],
            id_switch_sigma=config["ID_SWITCH_SIGMA"],
            id_switch_reference=config["ID_SWITCH_REFERENCE"],
            id_switch_subset_prob=config["ID_SWITCH_SUBSET_PROB"],
            id_switch_absent=config["ID_SWITCH_ABSENT"],
        ),
    ])


def get_image_hw(image: torch.Tensor | list | Image.Image):
    if isinstance(image, torch.Tensor):
        return image.shape[-2], image.shape[-1]
    elif isinstance(image, list):
        return get_image_hw(image[0])
    elif isinstance(image, Image.Image):
        return image.height, image.width
    else:
        raise NotImplementedError("The input image type is not supported.")
