# Vehicle tracking dataset (converted from CVAT video XML by tools/vehicles/prepare_vehicle_dataset.py).
#
# Layout (MOT-Challenge style, same as DanceTrack):
#   <DATA_ROOT>/Vehicles/{train,val,valmini}/<seq>/img1/00000001.jpg ...        (1-indexed frames)
#                                              /gt/gt.txt          TrackEval GT: class column = 1 for every vehicle
#                                                                  (class-agnostic HOTA), 8 = ignore regions
#                                              /gt/gt_classes.txt  same rows WITHOUT ignore regions, class column =
#                                                                  the real class index (0..4) -> detector training
#                                              /seqinfo.ini
#   <DATA_ROOT>/Vehicles/{train,val,valmini}_seqmap.txt
import os

from .dancetrack import DanceTrack
from .util import is_legal, append_annotation


class Vehicles(DanceTrack):
    def __init__(
            self,
            data_root: str = "./datasets/",
            sub_dir: str = "Vehicles",
            split: str = "train",
            load_annotation: bool = True,
    ):
        super().__init__(data_root=data_root, sub_dir=sub_dir, split=split, load_annotation=load_annotation)

    def _get_sequence_names(self):
        return sorted(d for d in os.listdir(os.path.join(self.data_dir, self.split))
                      if os.path.isdir(os.path.join(self.data_dir, self.split, d)))

    def _get_annotations(self):
        sequence_names = self._get_sequence_names()
        annotations = self._init_annotations(sequence_names)
        for sequence_name in sequence_names:
            sequence_dir = self._get_sequence_dir(self.data_dir, self.split, sequence_name)
            gt_file_path = os.path.join(sequence_dir, "gt", "gt_classes.txt")
            n_frames = self.sequence_infos[sequence_name]["length"]
            with open(gt_file_path, "r") as gt_file:
                for line in gt_file:
                    f = line.strip().split(",")
                    if len(f) < 8:
                        continue
                    frame_id, obj_id, category = int(f[0]), int(float(f[1])), int(float(f[7]))
                    x, y, w, h = map(float, f[2:6])
                    if not (1 <= frame_id <= n_frames) or w <= 0 or h <= 0:
                        continue
                    annotations[sequence_name][frame_id - 1] = append_annotation(
                        annotation=annotations[sequence_name][frame_id - 1],
                        obj_id=obj_id, category=category, bbox=[x, y, w, h], visibility=1.0,
                    )
            for i in range(n_frames):
                annotations[sequence_name][i]["is_legal"] = is_legal(annotations[sequence_name][i])
        return annotations
