#!/usr/bin/env python
"""Run train.py / submit_and_evaluate.py with REAL multi-process data parallelism on CPU (gloo) -- for testing the
distributed code paths (sampler sharding, DDP sync, EMA under DDP, metric gathering, sharded evaluation) without GPUs.
    torchrun --nproc_per_node 2 --master_addr 127.0.0.1 tools/ddp_cpu_launch.py train.py --config-path ... [flags]
(accelerate's own CLI does not start multi-process CPU runs unless PartialState(cpu=True) is created first.)"""
import os
import runpy
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from accelerate import PartialState

PartialState(cpu=True)
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
