import os
import sys
import atexit
import yaml
import logging
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import qdarktheme

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import qInstallMessageHandler, QtMsgType

from version import __version__
from gui.app import Master


LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


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
    log.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_tb))


sys.excepthook = handle_exception


def qt_message_handler(mode, _context, message):
    if mode == QtMsgType.QtDebugMsg:
        log.debug(f"Qt: {message}")
    elif mode == QtMsgType.QtInfoMsg:
        log.info(f"Qt: {message}")
    elif mode == QtMsgType.QtWarningMsg:
        log.warning(f"Qt: {message}")
    elif mode in (QtMsgType.QtCriticalMsg, QtMsgType.QtFatalMsg):
        log.critical(f"Qt: {message}")


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
