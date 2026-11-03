# Regression: metrics that exist on only some ranks must not deadlock `Metrics.sync()` (2 real processes, gloo).
import os
import signal
import subprocess
import sys
import textwrap

ROOT = os.path.join(os.path.dirname(__file__), "..")

SCRIPT = textwrap.dedent("""
    import os, sys
    sys.path.insert(0, os.environ["MOTIP_ROOT"])
    from accelerate import PartialState
    PartialState(cpu=True)
    import torch
    from log.log import Metrics
    rank = int(os.environ["RANK"])
    m = Metrics()
    m.update("always", float(rank + 1))
    if rank == 0:                      # only rank 0 ever sees this metric (e.g. a swap happened in its batch)
        m.update("only_rank0", 5.0)
        m.update("only_rank0", 7.0)
    m.sync()                           # used to hang: different number of collective calls per rank
    assert abs(m["always"].global_average - 1.5) < 1e-9, m["always"].global_average     # mean over both ranks
    assert abs(m["only_rank0"].global_average - 6.0) < 1e-9
    m2 = Metrics()
    m2.metrics["never_updated"]        # created but empty everywhere -> must not raise ZeroDivisionError
    m2.sync()
    assert m2["never_updated"].global_average != m2["never_updated"].global_average       # nan
    os.write(1, f"RANK{rank}_DONE\\n".encode())            # one atomic write: two ranks must not interleave characters
""")


def test_metrics_sync_with_rank_dependent_keys_does_not_deadlock(tmp_path):
    script = tmp_path / "w.py"
    script.write_text(SCRIPT)
    cmd = ["torchrun", "--nproc_per_node", "2", "--master_addr", "127.0.0.1", "--master_port", "29588", str(script)]
    p = subprocess.Popen(cmd, cwd=ROOT, env=dict(os.environ, MOTIP_ROOT=os.path.abspath(ROOT)), stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        out, _ = p.communicate(timeout=180)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        raise AssertionError("Metrics.sync() deadlocked with rank-dependent metric keys")
    assert p.returncode == 0, out[-800:]
    assert "RANK0_DONE" in out and "RANK1_DONE" in out
