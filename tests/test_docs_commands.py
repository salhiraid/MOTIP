# Documentation lint: every `train.py` / `submit_and_evaluate.py` / `tools/*.py` command shown in the docs (and every command
# emitted by tools/run_ablations.sh) must be accepted by the real argument parsers, and every flag must map to a config key.
import os
import re
import shlex
import subprocess
import sys

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

DOCS = ["README_MOTIP2.md", "docs/MOTIP2.md", "docs/VEHICLES.md", "docs/DATASETS.md"]


def extract_commands(md_path):
    text = open(os.path.join(ROOT, md_path)).read()
    cmds = []
    for block in re.findall(r"```(?:bash|shell|sh)\n(.*?)```", text, flags=re.S):
        block = re.sub(r"\\\n\s*", " ", block)                      # line continuations
        for line in block.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith(("for ", "done", "do ")):
                continue
            if re.search(r"(train\.py|submit_and_evaluate\.py|tools/[\w/]+\.py)", line):
                cmds.append(line.split("  #")[0].strip())
    return cmds


def tokens(cmd):
    cmd = re.sub(r"<[^>]+>", "PLACEHOLDER", cmd)
    cmd = re.sub(r"\$\w+|\$\{[^}]+\}", "PLACEHOLDER", cmd)
    t = shlex.split(cmd, comments=True)
    while t and re.match(r"^[A-Z_]+=", t[0]):                       # env prefixes (DRY_RUN=1 ...)
        t.pop(0)
    return t


def check_train_or_eval(argv):
    """argv = tokens after train.py / submit_and_evaluate.py"""
    import runtime_option
    from configs.util import load_super_config, update_config
    from utils.misc import yaml_to_dict
    old = sys.argv
    sys.argv = ["x"] + argv
    try:
        opt = runtime_option.runtime_option()                       # unknown flag => SystemExit
    finally:
        sys.argv = old
    cfg_path = opt.config_path
    if cfg_path and os.path.exists(os.path.join(ROOT, cfg_path.lstrip("./"))):
        cwd = os.getcwd()
        os.chdir(ROOT)
        try:
            cfg = yaml_to_dict(cfg_path)
            cfg = load_super_config(cfg, opt.super_config_path or cfg.get("SUPER_CONFIG_PATH"))
            update_config(config=cfg, option=opt)                    # RuntimeError if a flag has no config key
        finally:
            os.chdir(cwd)
    elif cfg_path and "PLACEHOLDER" not in cfg_path:
        raise AssertionError(f"config file does not exist: {cfg_path}")


def check_tool(script, argv):
    sub = [argv[0]] if script.endswith("prepare_dataset.py") and argv and not argv[0].startswith("-") else []   # subcommand flags
    help_out = subprocess.run([sys.executable, os.path.join(ROOT, script), *sub, "--help"], capture_output=True, text=True, cwd=ROOT)
    assert help_out.returncode == 0, f"{script} --help failed: {help_out.stderr[-300:]}"
    for a in argv:
        if a.startswith("--"):
            assert a.split("=")[0] in help_out.stdout, f"{script}: flag {a} is not accepted"


def check_command(cmd):
    t = tokens(cmd)
    if "train.py" in t:
        check_train_or_eval(t[t.index("train.py") + 1:])
    elif "submit_and_evaluate.py" in t:
        check_train_or_eval(t[t.index("submit_and_evaluate.py") + 1:])
    else:
        script = next((x for x in t if re.match(r"tools/[\w/]+\.py$", x)), None)
        if script:
            check_tool(script, t[t.index(script) + 1:])


@pytest.mark.parametrize("doc", DOCS)
def test_documented_commands_are_valid(doc):
    cmds = extract_commands(doc)
    assert cmds, f"no commands found in {doc}"
    for c in cmds:
        try:
            check_command(c)
        except (SystemExit, RuntimeError, AssertionError) as e:
            raise AssertionError(f"{doc}: invalid command\n  {c}\n  -> {type(e).__name__}: {e}")


@pytest.mark.parametrize("bench,extra", [("dancetrack", {}), ("sportsmot", {}), ("personpath22", {}), ("vehicles", {"INFER_DATASET": "Vehicles"})])
def test_run_ablations_emits_valid_commands(bench, extra):
    env = dict(os.environ, DRY_RUN="1", SIZE="S", BENCH=bench, NG="2", DATA="./datasets", **extra)
    out = subprocess.run(["bash", "tools/run_ablations.sh"], capture_output=True, text=True, cwd=ROOT, env=env)
    assert out.returncode == 0, out.stderr
    cmds = [l for l in out.stdout.splitlines() if l.startswith("accelerate launch")]
    assert len(cmds) >= 1 + 7 + 3 + 8                            # stage 1 + 7 retrains + N_init sweep + alpha sweep (4 x 2 modes)
    for c in cmds:
        check_command(c)


@pytest.mark.parametrize("bad", [
    "accelerate launch --num_processes=2 train.py --config-path configs/motip2/s_vehicles.yaml --not-a-real-flag 1",      # unknown flag
    "python train.py --config-path configs/r50_deformable_detr_motip_dancetrack.yaml --n-init 2",                          # legacy config has no N_INIT
    "python submit_and_evaluate.py --config-path configs/motip2/does_not_exist.yaml",                                      # missing file
    "python tools/idsw_stats.py --config x --data-root y --no-such-option 3",                                              # unknown tool flag
])
def test_the_lint_itself_rejects_bad_commands(bad):
    with pytest.raises((SystemExit, RuntimeError, AssertionError)):
        check_command(bad)


def test_the_lint_accepts_a_good_command():
    check_command("accelerate launch --num_processes=2 train.py --config-path configs/motip2/s_vehicles.yaml --n-init 2 --detr-class-indices 1 2 3 5 7")
