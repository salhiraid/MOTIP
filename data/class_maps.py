# Unified detector class sets shared by every converted dataset (data/converters/*) and by the configs.
#
# All datasets are converted to ONE class index space so that several of them can be mixed in a single training run
# (the DEIM detector has a single classification head; IDs stay class-agnostic).  The index of a class is its position
# in the tuple.  `vehicles5` is the class list of the user's own vehicle dataset (do not reorder it).
#
# COCO-80 rows (for initialising from the official COCO DEIM weights via --detr-class-indices) are given per set.
CLASS_SETS = {
    "vehicles5": ("bicycle", "car", "motorcycle", "bus", "truck"),
    "vehicles5_person": ("bicycle", "car", "motorcycle", "bus", "truck", "person"),
}
COCO80_ROWS = {"person": 0, "bicycle": 1, "car": 2, "motorcycle": 3, "bus": 5, "truck": 7}

IGNORE = "ignore"          # label meaning "region the evaluation must not penalise" (crowd / don't-care areas)
DISTRACTOR = "distractor"  # a real object of a class outside the set: never a training target, but a tracker output
                           # matched to it is not counted as a false positive (TrackEval class 8)


def class_names(class_set: str):
    if class_set not in CLASS_SETS:
        raise KeyError(f"unknown class set {class_set!r}; known: {sorted(CLASS_SETS)}")
    return CLASS_SETS[class_set]


def coco_rows(class_set: str):
    return [COCO80_ROWS[n] for n in class_names(class_set)]


def resolve_label(label_map: dict, label, class_set: str):
    """native label -> class index | IGNORE | DISTRACTOR | None (drop).  `label_map` maps native label -> unified class
    NAME or IGNORE / DISTRACTOR / None.  A name that is not in `class_set` becomes DISTRACTOR (e.g. `person` is a
    distractor for `vehicles5` but a real class for `vehicles5_person`)."""
    if label not in label_map:
        return None
    v = label_map[label]
    if v is None or v in (IGNORE, DISTRACTOR):
        return v
    names = class_names(class_set)
    if isinstance(v, int):
        return v
    return names.index(v) if v in names else DISTRACTOR


def default_max_interval(fps: float, cap: int = 4) -> int:
    """Largest random frame interval that keeps the time step comparable to DanceTrack (20 fps, interval <= 4 => 0.2 s):
    clip(round(fps / 5), 1, cap).  5 fps -> 1, 10 fps -> 2, 25/30 fps -> 4 (the paper's maximum)."""
    return int(max(1, min(cap, round(fps / 5.0))))
