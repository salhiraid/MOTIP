"""Drop-in replacement of the PyTorch MOTIP model for `models/runtime_tracker.RuntimeTracker` that runs the two ONNX graphs
written by tools/export_onnx.py with onnxruntime. Used to verify that the exported engines work inside the real tracking
loop (greedy readout, context buffer and tracklet confirmation stay in Python), and as a reference for a C++/TensorRT port.
    model = OnnxMOTIP("export/s"); tracker = RuntimeTracker(model=model, sequence_hw=(1080, 1920), ...)
The image passed to tracker.update must already have the exported detector resolution (H x W)."""
import json
import os
from types import SimpleNamespace

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn


class OnnxMOTIP(nn.Module):
    def __init__(self, export_dir, providers=("CPUExecutionProvider",)):
        super().__init__()
        self.meta = json.load(open(os.path.join(export_dir, "export_meta.json")))
        self.det = ort.InferenceSession(os.path.join(export_dir, self.meta["detector"]["file"]), providers=list(providers))
        self.dec = ort.InferenceSession(os.path.join(export_dir, self.meta["id_decoder"]["file"]), providers=list(providers))
        idm = self.meta["id_decoder"]
        # the attributes RuntimeTracker reads from the model
        self.detr = SimpleNamespace(hidden_dim=self.meta["detector"]["hidden_dim"])
        self.id_decoder = SimpleNamespace(spatial_anchor=idm["spatial_anchor"], set_rope_alpha=lambda *a, **k: None)
        self.num_id_vocabulary = idm["K"]
        self._names = idm["inputs"]
        self.expected_hw = tuple(self.meta["detector_resolution_HW"])

    @staticmethod
    def _np(t):
        return t.detach().cpu().numpy()

    def forward(self, frames=None, part=None, seq_info=None, **kw):
        if part == "detr":
            x = self._np(frames.tensors if hasattr(frames, "tensors") else frames).astype(np.float32)
            assert tuple(x.shape[-2:]) == self.expected_hw, f"detector was exported for {self.expected_hw}, got {x.shape[-2:]}"
            logits, boxes, emb, pos = self.det.run(None, {"x": x})
            return {"pred_logits": torch.from_numpy(logits), "pred_boxes": torch.from_numpy(boxes),
                    "outputs": torch.from_numpy(emb), "pos_embeds": torch.from_numpy(pos)}
        if part == "trajectory_modeling":
            return seq_info                                   # the adapter is part of id_decoder.onnx
        if part == "id_decoder":
            si = {k: v[0, 0] for k, v in seq_info.items() if torch.is_tensor(v)}
            T, N = si["trajectory_features"].shape[:2]
            M = si["unknown_features"].shape[2]
            K = self.num_id_vocabulary
            if N == 0 or T == 0:                              # nothing to associate with -> everything is newborn
                logits = torch.full((M, K + 1), -10.0)
                logits[:, K] = 10.0
                return logits[None, None, None], None, None
            feed = {
                "traj_feat": si["trajectory_features"], "traj_ids": si["trajectory_id_labels"].long(),
                "traj_times": si["trajectory_times"].long(), "traj_masks": si["trajectory_masks"],
                "unk_feat": si["unknown_features"][0], "unk_masks": si["unknown_masks"][0],
                "unk_times": si["unknown_times"][0].long(),
            }
            if self.id_decoder.spatial_anchor:
                feed["traj_pos"], feed["unk_pos"] = si["trajectory_pos"], si["unknown_pos"][0]
            out = self.dec.run(None, {n: self._np(feed[n]) for n in self._names})[0]
            return torch.from_numpy(out)[None, None, None], None, None
        raise NotImplementedError(part)
