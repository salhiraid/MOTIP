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
        self.single = "single" in self.meta                    # ONE motip2.onnx (detector + ID decoder) instead of two graphs
        if self.single:
            idm = self.meta["single"]
            self.net = ort.InferenceSession(os.path.join(export_dir, idm["file"]), providers=list(providers))
            self.det = self.dec = None
            self.tracker, self._cache = None, None
            hidden = idm["dim"]
        else:
            self.det = ort.InferenceSession(os.path.join(export_dir, self.meta["detector"]["file"]), providers=list(providers))
            self.dec = ort.InferenceSession(os.path.join(export_dir, self.meta["id_decoder"]["file"]), providers=list(providers))
            idm = self.meta["id_decoder"]
            hidden = self.meta["detector"]["hidden_dim"]
        # the attributes RuntimeTracker reads from the model
        self.detr = SimpleNamespace(hidden_dim=hidden)
        self.id_decoder = SimpleNamespace(spatial_anchor=idm["spatial_anchor"], set_rope_alpha=lambda *a, **k: None)
        self.num_id_vocabulary = idm["K"]
        self._names = idm["inputs"]
        self.expected_hw = tuple(self.meta["detector_resolution_HW"])

    @staticmethod
    def _np(t):
        return t.detach().cpu().numpy()

    def attach(self, tracker):
        """Single-graph mode: the detector call also needs the tracker's trajectory buffer (it is the same graph)."""
        self.tracker = tracker

    def _single_detr(self, x):
        tr = self.tracker
        assert tr is not None, "single-graph export: call model.attach(tracker) after building the RuntimeTracker"
        have = tr.trajectory_features.shape[0] > 0 and tr.trajectory_features.shape[1] > 0
        D = self.detr.hidden_dim
        if have:
            buf = {"traj_feat": tr.trajectory_features, "traj_pos": tr.trajectory_pos, "traj_ids": tr.trajectory_id_labels,
                   "traj_times": tr.trajectory_times, "traj_masks": tr.trajectory_masks}
        else:                                                   # first frame: dummy fully-masked buffer, id_logits are ignored
            buf = {"traj_feat": torch.zeros(2, 1, D), "traj_pos": torch.zeros(2, 1, D), "traj_ids": torch.zeros(2, 1, dtype=torch.long),
                   "traj_times": torch.arange(2)[:, None].long(), "traj_masks": torch.ones(2, 1, dtype=torch.bool)}
        feed = {"x": x, "det_thresh": np.array([tr.det_thresh], np.float32)}
        for n in self._names:
            if n in buf:
                v = self._np(buf[n])
                feed[n] = v.astype(np.int64) if n in ("traj_ids", "traj_times") else v
        feed = {n: feed[n] for n in self._names}
        return self.net.run(None, feed)

    def forward(self, frames=None, part=None, seq_info=None, **kw):
        if part == "detr" and self.single:
            x = self._np(frames.tensors if hasattr(frames, "tensors") else frames).astype(np.float32)
            assert tuple(x.shape[-2:]) == self.expected_hw, f"exported for {self.expected_hw}, got {x.shape[-2:]}"
            outs = self._single_detr(x)
            res = {"pred_logits": torch.from_numpy(outs[0]), "pred_boxes": torch.from_numpy(outs[1]),
                   "outputs": torch.from_numpy(outs[2]), "pos_embeds": torch.from_numpy(outs[3])}
            if len(outs) == 7:
                res["pred_keypoints"], res["pred_keypoint_vis"] = torch.from_numpy(outs[4]), torch.from_numpy(outs[5])
            self._cache = (res["outputs"][0], torch.from_numpy(outs[-1]))
            return res
        if part == "id_decoder" and self.single:
            uf = seq_info["unknown_features"][0, 0, 0]
            emb, logits = self._cache
            idx = (emb[None] == uf[:, None]).all(-1).float().argmax(-1)      # rows of the selected detections in the 300 queries
            return logits[idx][None, None, None], None, None
        if part == "detr":
            x = self._np(frames.tensors if hasattr(frames, "tensors") else frames).astype(np.float32)
            assert tuple(x.shape[-2:]) == self.expected_hw, f"detector was exported for {self.expected_hw}, got {x.shape[-2:]}"
            outs = self.det.run(None, {"x": x})
            logits, boxes, emb, pos = outs[:4]
            res = {"pred_logits": torch.from_numpy(logits), "pred_boxes": torch.from_numpy(boxes),
                   "outputs": torch.from_numpy(emb), "pos_embeds": torch.from_numpy(pos)}
            if len(outs) == 6:                                # detector exported with keypoints: xy (1,Q,K,2), visibility logits (1,Q,K)
                res["pred_keypoints"], res["pred_keypoint_vis"] = torch.from_numpy(outs[4]), torch.from_numpy(outs[5])
            return res
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
