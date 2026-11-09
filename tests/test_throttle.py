"""The shared throttle spaces live calls across processes (parallel benchmark passes)."""

import subprocess
import sys
import time

from toolhelper.mcp.throttle import SharedThrottle

CHILD = """
import sys, time
from toolhelper.mcp.throttle import SharedThrottle
t = SharedThrottle("pubmed", float(sys.argv[2]), sys.argv[1])
for _ in range(3):
    t.wait()
    print(time.time(), flush=True)
"""


def test_calls_in_one_process_are_spaced(tmp_path):
    t = SharedThrottle("pubmed", 0.2, tmp_path)
    assert t.wait() == 0  # nothing before it
    start = time.time()
    t.wait()
    assert time.time() - start >= 0.18 and (tmp_path / "pubmed.lock").exists()


def test_calls_from_two_processes_are_spaced(tmp_path):
    procs = [subprocess.Popen([sys.executable, "-c", CHILD, str(tmp_path), "0.25"], stdout=subprocess.PIPE, text=True)
             for _ in range(2)]
    stamps = sorted(float(line) for p in procs for line in p.communicate(timeout=60)[0].split())
    assert len(stamps) == 6
    assert min(b - a for a, b in zip(stamps, stamps[1:])) >= 0.22  # no two calls closer than the interval
