# MOTIP2 PLAN Phase 8 (decision #13): PersonPath22 loaders. The repo had none (only DanceTrack/SportsMOT/BFT/CrowdHuman),
# although submit_and_evaluate.py already referred to `PersonPath22_Inference` and bundles TrackEval's person_path_22.
#
# ASSUMED on-disk layout (MOT-Challenge style, i.e. PersonPath22 videos converted to frames + MOT gt.txt):
#   <DATA_ROOT>/PersonPath22/{train,val}/<seq>/{img1/00000001.jpg, gt/gt.txt, seqinfo.ini}      (training)
#   <DATA_ROOT>/PersonPath22_Inference/test/<seq>/{img1/00000001.jpg, seqinfo.ini}               (inference)
#   <DATA_ROOT>/PersonPath22_Inference/gts/person_path_22-test/<seq>/{gt/gt.txt, seqinfo.ini}    (TrackEval GT)
#   <DATA_ROOT>/PersonPath22_Inference/gts/seqmaps/person_path_22-test.txt                        (TrackEval seqmap)
# gt.txt columns: frame, id, x, y, w, h, conf, class, visibility (1-indexed frames); rows with conf == 0 are ignored.
import os

from .dancetrack import DanceTrack
from .util import is_legal, append_annotation


class PersonPath22(DanceTrack):
    def __init__(
            self,
            data_root: str = "./datasets/",
            sub_dir: str = "PersonPath22",
            split: str = "train",
            load_annotation: bool = True,
    ):
        super().__init__(data_root=data_root, sub_dir=sub_dir, split=split, load_annotation=load_annotation)

    def _get_annotations(self):
        sequence_names = self._get_sequence_names()
        annotations = self._init_annotations(sequence_names)
        for sequence_name in sequence_names:
            sequence_dir = self._get_sequence_dir(self.data_dir, self.split, sequence_name)
            with open(os.path.join(sequence_dir, "gt", "gt.txt"), "r") as gt_file:
                for line in gt_file:
                    fields = line.strip().split(",")
                    if len(fields) < 6:
                        continue
                    frame_id, obj_id = int(fields[0]), int(float(fields[1]))
                    x, y, w, h = map(float, fields[2:6])
                    conf = float(fields[6]) if len(fields) > 6 else 1.0
                    if conf == 0 or w <= 0 or h <= 0:        # MOT "ignore" rows / degenerate boxes
                        continue
                    annotations[sequence_name][frame_id - 1] = append_annotation(
                        annotation=annotations[sequence_name][frame_id - 1],
                        obj_id=obj_id, category=0, bbox=[x, y, w, h], visibility=1.0,
                    )
            for i in range(self.sequence_infos[sequence_name]["length"]):
                annotations[sequence_name][i]["is_legal"] = is_legal(annotations[sequence_name][i])
        return annotations


class PersonPath22_Inference(DanceTrack):
    """Inference-only view (no annotations): sequence infos + image paths of `<root>/PersonPath22_Inference/<split>`."""

    def __init__(
            self,
            data_root: str = "./datasets/",
            sub_dir: str = "PersonPath22_Inference",
            split: str = "test",
            load_annotation: bool = False,
    ):
        super().__init__(data_root=data_root, sub_dir=sub_dir, split=split, load_annotation=False)

    def _get_sequence_names(self):
        return sorted(d for d in os.listdir(os.path.join(self.data_dir, self.split))
                      if os.path.isdir(os.path.join(self.data_dir, self.split, d)))
