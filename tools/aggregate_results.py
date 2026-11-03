# MOTIP2 skill §11: "Report mean and standard deviation over at least 3 seeds for any claim of a gain under 1 HOTA."
# Aggregates the RESULTS.md run log. Tag a row's Notes with `variant=<name>` (tools/log_result.py --notes "variant=full").
#   python tools/aggregate_results.py [--results RESULTS.md]
import argparse
import re
import statistics
from collections import defaultdict

COLS = ["HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW"]


def parse(path):
    rows, header = [], None
    for line in open(path):
        if not line.startswith("|"):
            header = None if header and not line.strip() else header
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells[0] == "Date":
            header = cells
        elif header and len(cells) == len(header) and not set(cells[0]) <= set("-—"):
            rows.append(dict(zip(header, cells)))
    return rows


def aggregate(rows):
    groups = defaultdict(list)
    for r in rows:
        m = re.search(r"variant=([\w.\-]+)", r.get("Notes", ""))
        if m is None:
            continue
        groups[(r["Benchmark"], r["Split"], m.group(1))].append(r)
    out = {}
    for key, rs in groups.items():
        stats = {}
        for c in COLS:
            vals = [float(r[c]) for r in rs if r.get(c) not in (None, "", "—")]
            if vals:
                stats[c] = (statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else float("nan"))
        out[key] = {"n_seeds": len({r["Seed"] for r in rs}), "stats": stats}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="RESULTS.md")
    a = ap.parse_args()
    for (bench, split, variant), v in sorted(aggregate(parse(a.results)).items()):
        cols = "  ".join(f"{c} {m:.1f}±{s:.1f}" for c, (m, s) in v["stats"].items())
        flag = "" if v["n_seeds"] >= 3 else "   (<3 seeds: do not claim sub-1-HOTA gains)"
        print(f"{bench:14s} {split:5s} {variant:16s} n={v['n_seeds']}  {cols}{flag}")
