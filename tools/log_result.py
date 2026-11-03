# MOTIP2 PLAN Phase 1 / skill §15: append one row to RESULTS.md from a TrackEval summary.
# Usage:
#   python tools/log_result.py --summary outputs/<run>/evaluate/default/DanceTrack/val/<ckpt>/tracker/pedestrian_summary.txt \
#       --config configs/motip2/s_dancetrack.yaml --seed 42 --benchmark DanceTrack --split val \
#       --checkpoint <path> [--latency 27.1] [--notes "..."]
import argparse
import datetime
import subprocess

HEADER_MARK = "| Date | Git commit |"


def read_summary(path: str) -> dict:
    with open(path) as f:
        names = f.readline().strip().split(" ")
        values = f.readline().strip().split(" ")
    return {n: float(v) for n, v in zip(names, values)}


def build_row(args, m: dict) -> str:
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()

    def f(k, fmt="{:.1f}"):
        return fmt.format(m[k]) if k in m else "—"

    cells = [
        datetime.date.today().isoformat(), commit, args.config, str(args.seed), args.benchmark, args.split,
        f("HOTA"), f("DetA"), f("AssA"), f("MOTA"), f("IDF1"), f("IDSW", "{:.0f}"),
        args.latency or "—", args.checkpoint or "—", args.notes or "",
    ]
    return "| " + " | ".join(cells) + " |"


def append_row(results_path: str, row: str):
    with open(results_path) as f:
        lines = f.read().split("\n")
    # Insert after the last table line that follows the run-log header.
    start = next(i for i, l in enumerate(lines) if l.startswith(HEADER_MARK))
    end = start + 2
    while end < len(lines) and lines[end].startswith("|"):
        end += 1
    # Drop the placeholder row if present.
    body = [l for l in lines[start + 2:end] if not l.startswith("| — | — | — |")]
    lines[start + 2:end] = body + [row]
    with open(results_path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--summary", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--benchmark", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--latency", default=None)
    p.add_argument("--notes", default=None)
    p.add_argument("--results", default="RESULTS.md")
    a = p.parse_args()
    row = build_row(a, read_summary(a.summary))
    append_row(a.results, row)
    print(row)
