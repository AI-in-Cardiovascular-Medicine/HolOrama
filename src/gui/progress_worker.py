"""Background-thread runner that forwards stdout lines as Qt signals, so long blocking steps
(CCTA centerline extraction, fusion remeshing) can show their print() output live."""

import sys

from PyQt6.QtCore import QThread, pyqtSignal


class StdoutCapturingWorker(QThread):
    """Runs `fn(*args, **kwargs)` off the main thread, emits line_printed per stdout line,
    then finished_ok(result) or failed(message). Redirects sys.stdout process-wide while
    running, so only one such worker may run at a time."""

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
