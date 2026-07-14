import atexit
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from types import FrameType, SimpleNamespace

import yaml
from loguru import logger

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import qdarktheme
from PyQt6.QtCore import QtMsgType, qInstallMessageHandler
from PyQt6.QtWidgets import QApplication

from gui.app import Master
from version import __version__

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"


class _InterceptHandler(logging.Handler):
    """Forwards stdlib `logging` records (h5py, matplotlib, Qt, ...) into loguru's
    sinks so everything ends up in one file with one format instead of two competing
    logging configs writing to the same path."""

    def emit(self, record: logging.LogRecord) -> None:
        level: str | int
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Walk past the stdlib logging frames so loguru attributes {name} to the
        # original caller (e.g. "h5py._conv") instead of this handler.
        frame: FrameType | None = logging.currentframe()
        depth = 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


# WARNING here (not DEBUG) so third-party libraries' routine DEBUG/INFO chatter
# (e.g. h5py._conv "Creating converter from X to Y") never reaches loguru at all.
logging.basicConfig(handlers=[_InterceptHandler()], level=logging.WARNING)

logger.remove()  # drop loguru's default stderr sink so console output is controlled below
logger.add(LOG_FILE, level="ERROR", format="{time:YYYY-MM-DD HH:mm:ss} [{level}] {name}: {message}")
logger.add(sys.stdout, level="WARNING", format="{time:YYYY-MM-DD HH:mm:ss} [{level}] {name}: {message}")


def _cleanup_empty_log():
    logging.shutdown()
    if LOG_FILE.exists() and LOG_FILE.stat().st_size == 0:
        LOG_FILE.unlink()


atexit.register(_cleanup_empty_log)


def handle_exception(exc_type, exc_value, exc_tb):
    """Catch any uncaught exception and log it before the app dies."""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_tb)
        return
    logger.opt(exception=(exc_type, exc_value, exc_tb)).critical("Uncaught exception")


sys.excepthook = handle_exception


def qt_message_handler(mode, _context, message):
    if mode == QtMsgType.QtDebugMsg:
        logger.debug(f"Qt: {message}")
    elif mode == QtMsgType.QtInfoMsg:
        logger.info(f"Qt: {message}")
    elif mode == QtMsgType.QtWarningMsg:
        logger.warning(f"Qt: {message}")
    elif mode in (QtMsgType.QtCriticalMsg, QtMsgType.QtFatalMsg):
        logger.critical(f"Qt: {message}")


qInstallMessageHandler(qt_message_handler)


def _print_banner():
    print(
        r"""
                )                        
    )     (  ( /(                        
 ( /(     )\ )\()) (      )    )      )  
 )\()) ( ((_|(_)\  )(  ( /(   (    ( /(  
((_)\  )\ _   ((_)(()\ )(_))  )\  ')(_)) 
| |(_)((_) | / _ \ ((_|(_)_ _((_))((_)_  
| ' \/ _ \ || (_) | '_/ _` | '  \() _` | 
|_||_\___/_| \___/|_| \__,_|_|_|_|\__,_| 
                                         
       """
    )
    print(f"  version  : {__version__}")
    print("  docs     : https://aivus-caa.readthedocs.io")
    print("  license  : MIT")
    print("  author   : yungselm\n")


if os.environ.get("AIVUS_SILENT", "0") == "0":
    _print_banner()


def _load_config(path: Path) -> SimpleNamespace:
    def _to_ns(obj):
        if isinstance(obj, dict):
            return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
        return obj

    with open(path, encoding="utf-8") as f:
        return _to_ns(yaml.safe_load(f))


def main() -> None:
    config = _load_config(Path(__file__).parent / 'config.yaml')
    app = QApplication(sys.argv)
    app.setApplicationVersion(__version__)

    qdarktheme.setup_theme('dark')  # switch to auto to recognize system mode
    _window = Master(config)

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
