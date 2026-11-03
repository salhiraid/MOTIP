#!/usr/bin/env python
"""Convert a public tracking dataset to the unified MOT layout that `data/unified_mot.py` trains on.

    python tools/prepare_dataset.py <dataset> --root <downloaded dataset> --out ./datasets [options]

datasets: bdd100k detrac visdrone uavdt kitti cityflow fasttracker (any MOT-format data) waymo nuscenes argoverse2 boxes3d
Run with --dry-run first: it reads the annotations only and prints class/size/ID statistics and the recommended
NUM_ID_VOCABULARY.  See docs/DATASETS.md for per-dataset details, licences and caveats.
"""
import argparse
import importlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

MODULES = {"bdd100k": "bdd100k", "detrac": "detrac", "visdrone": "visdrone", "uavdt": "uavdt", "kitti": "kitti",
           "cityflow": "cityflow", "fasttracker": "mot_format", "mot": "mot_format", "waymo": "waymo",
           "nuscenes": "nuscenes", "argoverse2": "argoverse2", "boxes3d": "boxes3d_generic"}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="dataset", required=True)
    mods = {}
    from data.converters import common
    for name, mod in MODULES.items():
        m = importlib.import_module(f"data.converters.{mod}")
        mods[name] = m
        sp = sub.add_parser(name, help=(m.__doc__ or "").strip().splitlines()[0])
        common.add_common_args(sp)
        m.add_args(sp)
        sp.add_argument("--dir-name", default=None, help=f"output folder name (default {m.DIR})")
    a = ap.parse_args(argv)
    m = mods[a.dataset]
    dir_name = a.dir_name or m.DIR
    if a.dataset == "mot" and not a.dir_name:
        raise SystemExit("--dir-name is required for the generic 'mot' converter")
    class_map = common.load_class_map(m.DEFAULT_MAP, a.class_map)
    print(f"{a.dataset}: licence note: {m.LICENSE}" if hasattr(m, "LICENSE") else a.dataset)
    names = common.class_names(a.class_set)
    ann = getattr(m, "ANNOTATED", None)          # [] = class-less/coarse labels, "data" = classes present, None = from the map
    if ann is None:
        ann = sorted({v for v in class_map.values() if isinstance(v, str) and v in names}, key=names.index)
    return common.run_conversion(dir_name, m.sequences(a, class_map, a.class_set), a, a.class_set,
                                 notes=[getattr(m, "LICENSE", "")], annotated=ann)


if __name__ == "__main__":
    main()
