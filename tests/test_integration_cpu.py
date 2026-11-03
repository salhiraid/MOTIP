# End-to-end integration on CPU with freshly generated synthetic videos: the real entry points (train.py, submit_and_evaluate.py)
# in the order a user runs them, including resume, submit mode and REAL 2-process data-parallel training (gloo).
import os
import re
import shutil
import subprocess
import sys

import pytest
import torch

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools", "vehicles"))

pytestmark = [pytest.mark.slow, pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg missing")]


def sh(cmd, env=None, timeout=1500):
    e = dict(os.environ, **(env or {}))
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, env=e, timeout=timeout)
    out = re.sub(r"\x1b\[[0-9;]*m", "", r.stdout + r.stderr)
    assert r.returncode == 0, out[-1500:]
    return out


def metrics(out, tag):
    line = [l for l in out.splitlines() if tag in l][-1]
    return {k: float(v) for k, v in re.findall(r"(\w+) = (-?[0-9.]+)", line)}


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    import make_synthetic_vehicle_videos as G
    import prepare_vehicle_dataset as P
    from models.deim.builder import build_deim
    d = tmp_path_factory.mktemp("integ")
    G.main(["--out", str(d / "src"), "--videos-per-site", "2", "--frames", "60"])
    P.main(["--csv", str(d / "src" / "dataset.csv"), "--videos-dir", str(d / "src"), "--out", str(d / "data"), "--val-frac", "0.3",
            "--resize-short", "192", "--max-long", "320", "--valmini", "2", "--workers", "2", "--min-frames", "20"])
    torch.manual_seed(0)
    torch.save({"model": build_deim("N", num_classes=5).state_dict()}, d / "vehicle_detector_5cls.pth")      # a "converted" checkpoint
    torch.save({"model": build_deim("N", num_classes=80).state_dict()}, d / "coco80.pth")
    return d


def train(world, stage, pretrain, name, extra=(), launcher=("python",)):
    cfg = f"configs/motip2/smoke_vehicles_stage{stage}.yaml"
    return sh([*launcher, "train.py", "--config-path", cfg, "--data-root", str(world / "data"), "--detr-pretrain", str(pretrain),
               "--exp-name", name, "--outputs-dir", str(world / name), *extra])


def s1(world):
    """Stage-1 checkpoint, built on first use (tests must not depend on execution order)."""
    ck = world / "s1" / "checkpoint_0.pth"
    if not ck.exists():
        train(world, 1, world / "vehicle_detector_5cls.pth", "s1")
    return ck


def s2(world):
    ck = world / "s2" / "checkpoint_0.pth"
    if not ck.exists():
        train(world, 2, s1(world), "s2", ["--epochs", "1", "--samples-per-epoch", "16"])
    return ck


def test_stage1_from_a_converted_5_class_checkpoint_and_from_coco_rows(world):
    out = train(world, 1, world / "vehicle_detector_5cls.pth", "s1")
    assert "Finish epoch" in out and "Eval epoch" in out
    ck = torch.load(world / "s1" / "checkpoint_0.pth", weights_only=False)
    assert ck["ema"]["updates"] > 0 and any("score_head" in k for k in ck["model"])                       # EMA weights are saved
    out = train(world, 1, world / "coco80.pth", "s1coco", ["--detr-class-indices", "1", "2", "3", "5", "7"])
    assert "Finish epoch" in out


def test_stage2_all_priors_logs_losses_and_live_idswitch_statistics(world):
    out = train(world, 2, s1(world), "s2", ["--epochs", "1", "--samples-per-epoch", "16"])
    m = metrics(out, "Finish epoch")
    assert {"id_loss", "id_ce", "id_giou", "id_l1", "detr_loss", "idsw_swap_pct"} <= set(m)
    assert m["id_loss"] == pytest.approx(m["id_ce"] + 2 * m["id_giou"] + 5 * m["id_l1"], rel=2e-2)      # 1*CE + 2*GIoU + 5*L1
    ck = torch.load(world / "s2" / "checkpoint_0.pth", weights_only=False)
    assert ck["ema"]["updates"] == 8 and ck["states"]["global_step"] == 8


def evaluate(world, ckpt, name, extra=()):
    return sh(["python", "submit_and_evaluate.py", "--config-path", "configs/motip2/smoke_vehicles_stage2.yaml", "--data-root",
               str(world / "data"), "--inference-mode", "evaluate", "--inference-model", str(ckpt), "--inference-dataset", "Vehicles",
               "--inference-split", "val", "--outputs-dir", str(world / name), *extra])


def test_evaluation_produces_tracks_and_detector_only_mode_ignores_n_init(world):
    out = evaluate(world, s2(world), "ev", ["--n-init", "1"])
    m = metrics(out, "Finish evaluation")
    assert "HOTA" in m and m["CLR_FP"] + m["CLR_FN"] > 0
    files = list((world / "ev" / "evaluate" / "default" / "Vehicles" / "val" / "checkpoint_0" / "tracker").glob("v*.txt"))
    assert files and sum(len(f.read_text().splitlines()) for f in files) > 0                                 # tracks were written
    # detector-only evaluation of a stage-1 checkpoint must output boxes although N_INIT=2 is the default
    out = sh(["python", "submit_and_evaluate.py", "--config-path", "configs/motip2/smoke_vehicles_stage1.yaml", "--data-root",
              str(world / "data"), "--inference-mode", "evaluate", "--inference-model", str(s1(world)),
              "--inference-dataset", "Vehicles", "--inference-split", "val", "--outputs-dir", str(world / "ev_det")])
    assert metrics(out, "Finish evaluation")["CLR_FP"] > 0


def test_submit_mode_writes_tracker_files_without_ground_truth_scoring(world):
    sh(["python", "submit_and_evaluate.py", "--config-path", "configs/motip2/smoke_vehicles_stage2.yaml", "--data-root",
        str(world / "data"), "--inference-mode", "submit", "--inference-model", str(s2(world)),
        "--inference-dataset", "Vehicles", "--inference-split", "val", "--outputs-dir", str(world / "sub"), "--n-init", "1"])
    assert list((world / "sub" / "submit" / "default" / "Vehicles" / "val" / "checkpoint_0" / "tracker").glob("v*.txt"))


def test_resume_continues_epoch_step_and_ema_counters(world):
    out = train(world, 2, s1(world), "s2r", ["--epochs", "2", "--samples-per-epoch", "16", "--resume-model", str(s2(world))])
    assert "Start from epoch 1, step 8" in out
    ck = torch.load(world / "s2r" / "checkpoint_1.pth", weights_only=False)
    assert ck["states"]["global_step"] == 16 and ck["ema"]["updates"] == 16


def test_real_two_process_data_parallel_training_and_sharded_evaluation(world):
    launcher = ["torchrun", "--nproc_per_node", "2", "--master_addr", "127.0.0.1", "--master_port", "29577", "tools/ddp_cpu_launch.py"]
    out = train(world, 2, s1(world), "ddp", ["--epochs", "1", "--samples-per-epoch", "16"], launcher=launcher)
    assert "DistributedType.MULTI_CPU" in out
    assert "Finish epoch" in out and "Eval epoch" in out                       # DDP step + EMA + sharded evaluation + metric gathering
    ck = torch.load(world / "ddp" / "checkpoint_0.pth", weights_only=False)
    assert ck["ema"]["updates"] == 4                                           # 16 samples / (2 clips x 2 processes)
