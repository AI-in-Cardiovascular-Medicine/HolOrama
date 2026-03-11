import os
import sys
import hydra
import logging
from pathlib import Path
from datetime import datetime

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import qdarktheme

from omegaconf import DictConfig
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import qInstallMessageHandler, QtMsgType

from version import __version__
from gui.gui import Master

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)


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
    print(r"""
      {_       {__{__         {__{__     {__  {__ __
     {_ __     {__ {__       {__ {__     {__{__    {__
    {_  {__    {__  {__     {__  {__     {__ {__
   {__   {__   {__   {__   {__   {__     {__   {__
  {______ {__  {__    {__ {__    {__     {__      {__
 {__       {__ {__     {____     {__     {__{__    {__
{__         {__{__      {__        {_____     {__ __
""")
    print(f"  version  : {__version__}")
    print(f"  docs     : https://aivus-caa.readthedocs.io")
    print(f"  license  : MIT\n")


if os.environ.get("AIVUS_SILENT", "0") == "0":
    _print_banner()


@hydra.main(version_base=None, config_path='.', config_name='config')
def main(config: DictConfig) -> None:
    app = QApplication(sys.argv)
    app.setApplicationVersion(__version__)

    qdarktheme.setup_theme('dark')  # switch to auto to recognize system mode
    _window = Master(config)

    sys.exit(app.exec())


if __name__ == '__main__':
    main()