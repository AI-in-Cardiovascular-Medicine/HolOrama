"""Background-thread runner that forwards stdout lines as Qt signals.

Every other pipeline step in this page is either fast enough not to need live progress,
or reports it through multimodars' own return values. fix_and_remesh_stitched_mesh is the
one exception (pure Python, no Rust, several seconds, and its verbose=True output goes to
print() rather than a callback) — this lets the "Fix && Remesh" progress dialog show that
output live instead of a static "please wait" message.
"""

import sys

from PyQt6.QtCore import QThread, pyqtSignal


class StdoutCapturingWorker(QThread):
    """Runs `fn(*args, **kwargs)` off the main thread; emits line_printed for every line
    it writes to stdout, then exactly one of finished_ok(result) / failed(message).

    Note: sys.stdout is process-global, so this redirects it for the whole process while
    the thread runs, not just for this call. Fine here since the app only ever has one
    such blocking step in flight at a time (driven by an explicit button click).
    """

    line_printed = pyqtSignal(str)
    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, fn, args: tuple, kwargs: dict, parent=None) -> None:
        super().__init__(parent)
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    def run(self) -> None:
        original_stdout = sys.stdout
        sys.stdout = _LineEmitter(self.line_printed.emit)
        try:
            result = self._fn(*self._args, **self._kwargs)
        except Exception as e:
            self.failed.emit(str(e))
        else:
            self.finished_ok.emit(result)
        finally:
            sys.stdout = original_stdout


class _LineEmitter:
    """Minimal file-like object: buffers partial writes, emits one signal per full line."""

    def __init__(self, emit) -> None:
        self._emit = emit
        self._buffer = ''

    def write(self, text: str) -> None:
        self._buffer += text
        while '\n' in self._buffer:
            line, self._buffer = self._buffer.split('\n', 1)
            if line.strip():
                self._emit(line)

    def flush(self) -> None:
        pass
