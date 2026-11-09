"""Spacing between live calls to one upstream, shared by every process on the machine.

Experiments run several benchmark passes at once. A per-process limit let three passes send
PubMed E-utilities about three times its 3 requests/s: p10 logged 298 HTTP 429s, which Medical
MCP reports as "No medical articles found", so the recorder saved them as real empty results.
The time of the last call lives in a small file in the temp directory; a process holds a lock on
it while it waits for its turn, so calls from all processes are spaced by `interval`.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

DEFAULT_DIR = Path(tempfile.gettempdir()) / "toolhelper-throttle"
STAMP_BYTES = 32
LOCK_OFFSET = 4096  # the byte locked as a mutex, past the timestamp


def _try_lock(fd: int) -> bool:
    os.lseek(fd, LOCK_OFFSET, os.SEEK_SET)
    try:
        if sys.platform == "win32":
            import msvcrt

            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.lockf(fd, fcntl.LOCK_EX | fcntl.LOCK_NB, 1, LOCK_OFFSET)
    except OSError:
        return False
    return True


def _unlock(fd: int) -> None:
    os.lseek(fd, LOCK_OFFSET, os.SEEK_SET)
    if sys.platform == "win32":
        import msvcrt

        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.lockf(fd, fcntl.LOCK_UN, 1, LOCK_OFFSET)


class SharedThrottle:
    def __init__(self, name: str, interval: float, directory: Path | str | None = None) -> None:
        self.path = Path(directory or DEFAULT_DIR) / f"{name}.lock"
        self.interval = interval

    def wait(self) -> float:
        """Blocks until `interval` seconds after the last call by any process, then records this
        call. Returns the seconds waited."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT)
        try:
            while not _try_lock(fd):
                time.sleep(0.02)
            try:
                os.lseek(fd, 0, os.SEEK_SET)
                raw = os.read(fd, STAMP_BYTES).strip()
                wait = max(0.0, (float(raw) if raw else 0.0) + self.interval - time.time())
                if wait:
                    time.sleep(wait)
                os.lseek(fd, 0, os.SEEK_SET)
                os.write(fd, f"{time.time():.6f}".ljust(STAMP_BYTES).encode())
                return wait
            finally:
                _unlock(fd)
        finally:
            os.close(fd)
