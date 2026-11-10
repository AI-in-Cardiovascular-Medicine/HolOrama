"""Run a command with its output streamed live, for pixi and the vmtk steps."""

from __future__ import annotations

import os
import subprocess
import threading
import time
from collections.abc import Callable

_HEARTBEAT_SECONDS = 15  # how often to print "still running" while a step is silent
# No console window flashing up for every step when HolOrama runs as a windowed app.
_NO_WINDOW = getattr(subprocess, 'CREATE_NO_WINDOW', 0)


def run_streaming(cmd: list[str], log_cb: Callable[[str], None] | None = None) -> tuple[int, str]:
    """Run cmd, streaming its combined stdout and stderr to log_cb line by line as it runs.
    vmtk's centerline extraction can stay silent for a long time, so a heartbeat line every
    _HEARTBEAT_SECONDS of silence tells slow from frozen. Returns (exit code, full output)."""
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1,
            creationflags=_NO_WINDOW,
            # Plain line-by-line output: no ANSI colours or progress bars in the progress dialog.
            env={**os.environ, 'PIXI_COLOR': 'never', 'PIXI_NO_PROGRESS': 'true', 'NO_COLOR': '1'},
        )
    except OSError as e:
        raise RuntimeError(f'Could not start {cmd[0]}: {e}') from e

    heartbeat = _Heartbeat(log_cb)
    heartbeat.start()
    lines: list[str] = []
    assert process.stdout is not None
    try:
        for line in process.stdout:
            line = line.rstrip('\n')
            lines.append(line)
            heartbeat.output_seen()
            if log_cb is not None:
                # pixi prints symbols such as a check mark. A Windows console (cp1252) cannot
                # encode them and print() would raise mid-run.
                log_cb(line.encode('ascii', 'replace').decode('ascii'))
    finally:
        heartbeat.stop()
    process.wait()
    return process.returncode, '\n'.join(lines)


class _Heartbeat(threading.Thread):
    """Logs a 'still running' line whenever the command printed nothing for _HEARTBEAT_SECONDS."""

    def __init__(self, log_cb: Callable[[str], None] | None) -> None:
        super().__init__(daemon=True)
        self._log_cb = log_cb
        self._start = time.monotonic()
        self._last_output = self._start
        self._stopped = threading.Event()

    def output_seen(self) -> None:
        self._last_output = time.monotonic()

    def stop(self) -> None:
        self._stopped.set()
        self.join(timeout=1.0)

    def run(self) -> None:
        while not self._stopped.wait(1.0):
            now = time.monotonic()
            if self._log_cb is not None and now - self._last_output >= _HEARTBEAT_SECONDS:
                self._log_cb(f'    ...still running ({int(now - self._start)}s elapsed, no output yet)')
                self._last_output = now
