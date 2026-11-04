#!/usr/bin/env python
"""Convert a public tracking dataset to the unified MOT layout (thin launcher for data/converters/prepare.py).

    python tools/prepare_dataset.py <dataset> --root <downloaded dataset> --out ./datasets [options]
    python tools/prepare_dataset.py <dataset> --help          # per-dataset options

datasets: bdd100k detrac visdrone uavdt kitti cityflow fasttracker mot waymo nuscenes argoverse2 boxes3d.
Always run with --dry-run first.  Details, licences and caveats: docs/DATASETS.md
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from data.converters.prepare import main   # noqa: E402

if __name__ == "__main__":
    main()
