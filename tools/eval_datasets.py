#!/usr/bin/env python
"""Evaluate ONE checkpoint on several datasets/splits (submit_and_evaluate.py once per dataset) and print a table.

    python tools/eval_datasets.py --config configs/motip2/mixes/mix_a.yaml --model outputs/mix_a/checkpoint_9.pth \
        --datasets Vehicles:val BDD100K_MOT:val UA_DETRAC:val --outputs-dir outputs/mix_a [--run] [--gpus 4]

Without --run it only prints the commands.  HOTA/DetA/AssA/MOTA/IDF1/IDSW are read from the TrackEval summaries
(`pedestrian_summary.txt`; HOTA is class-agnostic, see docs/DATASETS.md).  Each dataset is evaluated on its own
sequences with its own ignore regions; averaging across datasets is up to you (sizes differ by orders of magnitude).
"""
import argparse
import os
import subprocess
import sys

METRICS = ["HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW"]


def command(a, name, split):
    return ["accelerate", "launch", f"--num_processes={a.gpus}", "submit_and_evaluate.py", "--data-root", a.data_root,
            "--config-path", a.config, "--inference-mode", "evaluate", "--inference-model", a.model,
            "--inference-dataset", name, "--inference-split", split, "--outputs-dir", a.outputs_dir, *a.extra]


def summary_path(a, name, split):
    model = os.path.basename(a.model)[:-4]
    return os.path.join(a.outputs_dir, "evaluate", a.group, name, split, model, "tracker", "pedestrian_summary.txt")


def read_summary(path):
    with open(path) as f:
        names, values = f.readline().split(), f.readline().split()
    return {n: float(v) for n, v in zip(names, values)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--datasets", nargs="+", required=True, help="Name:split ...")
    ap.add_argument("--outputs-dir", required=True)
    ap.add_argument("--data-root", default="./datasets")
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--group", default="default")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("extra", nargs="*", help="extra args for submit_and_evaluate.py (after --)")
    a = ap.parse_args(argv)
    rows = []
    for item in a.datasets:
        name, _, split = item.partition(":")
        split = split or "val"
        cmd = command(a, name, split)
        print(" ".join(cmd), flush=True)
        if a.run:
            r = subprocess.run(cmd)
            if r.returncode != 0:
                print(f"!! evaluation of {name}:{split} failed (exit {r.returncode})", file=sys.stderr)
                rows.append((name, split, None))
                continue
        p = summary_path(a, name, split)
        rows.append((name, split, read_summary(p) if os.path.exists(p) else None))
    if any(r[2] for r in rows):
        print("\n| dataset | split | " + " | ".join(METRICS) + " |\n|---|---|" + "---|" * len(METRICS))
        for name, split, m in rows:
            print(f"| {name} | {split} | " + " | ".join(f"{m[k]:.1f}" if m and k in m else "-" for k in METRICS) + " |")
    return rows


if __name__ == "__main__":
    main()
