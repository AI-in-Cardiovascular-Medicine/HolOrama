import atexit
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from types import FrameType, SimpleNamespace

import yaml
from loguru import logger

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import qdarktheme
from PyQt6.QtCore import QtMsgType, qInstallMessageHandler
from PyQt6.QtGui import QFontDatabase
from PyQt6.QtWidgets import QApplication

from gui.app import Master
from version import __version__

# When frozen (Nuitka standalone), the app may be installed under a read-only
# location such as C:\Program Files, and the shortcut's working directory points
# there. The only two things the app writes on its own — its logs and its config
# file — must therefore live in a per-user, always-writable directory instead of
# next to the exe / relative to the CWD (which raises PermissionError on startup).
# User data (contours, reports, NIfTi/STL exports) is unaffected: it keeps writing
# next to the opened data file. Uncompiled dev runs keep the original in-repo paths.
IS_FROZEN = "__compiled__" in globals()


def _user_data_dir() -> Path:
    base = Path(os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA") or Path.home())
    return base / "HolOrama"


LOG_DIR = (_user_data_dir() / "logs") if IS_FROZEN else Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
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
logger.add(LOG_FILE, level="WARNING", format="{time:YYYY-MM-DD HH:mm:ss} [{level}] {name}: {message}")
logger.add(sys.stdout, level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} [{level}] {name}: {message}")


def _cleanup_empty_log():
    logging.shutdown()
    logger.remove()  # close loguru's own file sink — otherwise it still holds LOG_FILE
    # open on Windows and unlink() below fails with PermissionError (WinError 32).
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
    print("  docs     : https://holorama.readthedocs.io")
    print("  license  : MIT")
    print("  author   : Anselm W. Stark <anselm.stark@insel.ch>\n")


if os.environ.get("AIVUS_SILENT", "0") == "0":
    _print_banner()


def _load_config(path: Path) -> SimpleNamespace:
    def _to_ns(obj):
        if isinstance(obj, dict):
            return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
        return obj

    with open(path, encoding="utf-8") as f:
        config = _to_ns(yaml.safe_load(f))
    config._config_path = path
    return config


def _merge_missing(bundled: dict, user: dict) -> bool:
    """Recursively add every key present in `bundled` but missing from `user`, leaving
    values the user already set untouched. Returns True if anything was added."""
    changed = False
    for key, value in bundled.items():
        if key not in user:
            user[key] = value
            changed = True
        elif isinstance(value, dict) and isinstance(user.get(key), dict):
            changed |= _merge_missing(value, user[key])
    return changed


def _migrate_user_config(bundled: Path, user_cfg: Path) -> None:
    """Bring an existing per-user config.yaml up to the bundled schema.

    The per-user copy is seeded once and then survives every upgrade, so a file written
    by an older version lacks any section or key added since - and the app reads those
    unguarded (e.g. config.intravascular.n_points_contour), so it dies at startup with
    AttributeError rather than falling back to a default. Merge the bundled defaults in
    for whatever the user file is missing; values the user set win. The previous copy is
    kept as config.yaml.bak, and one we cannot parse is replaced wholesale."""
    from ruamel.yaml import YAML

    yaml_rt = YAML(typ='rt')
    yaml_rt.preserve_quotes = True
    yaml_rt.boolean_representation = ['False', 'True']  # type: ignore[attr-defined]
    yaml_rt.indent(mapping=2, sequence=4, offset=2)
    backup = user_cfg.with_name('config.yaml.bak')

    try:
        with open(user_cfg, encoding='utf-8') as f:
            user_data = yaml_rt.load(f)
        with open(bundled, encoding='utf-8') as f:
            bundled_data = yaml_rt.load(f)
    except Exception as exc:  # corrupt / unparseable user config - start over from the default
        logger.warning(f'Could not read {user_cfg} ({exc}); keeping it as {backup.name} and reseeding')
        shutil.copyfile(user_cfg, backup)
        shutil.copyfile(bundled, user_cfg)
        return

    if not isinstance(user_data, dict):
        shutil.copyfile(user_cfg, backup)
        shutil.copyfile(bundled, user_cfg)
        return

    if _merge_missing(bundled_data, user_data):
        shutil.copyfile(user_cfg, backup)
        with open(user_cfg, 'w', encoding='utf-8') as f:
            yaml_rt.dump(user_data, f)
        logger.info(f'Added new default settings to {user_cfg} (previous copy kept as {backup.name})')


def _resolve_config_path() -> Path:
    """Return the config.yaml to load. In dev this is the copy next to the source; in
    the frozen app it must be a writable per-user copy so 'Display Settings...' can save
    back to it (the bundled one may sit under read-only C:\\Program Files). The per-user
    copy is seeded from the bundled default on first run, and merged with it on later
    runs so settings a newer version added reach a config file an older version wrote."""
    bundled = Path(__file__).parent / 'config.yaml'
    if not IS_FROZEN:
        return bundled
    user_cfg = _user_data_dir() / 'config.yaml'
    if not user_cfg.exists():
        user_cfg.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(bundled, user_cfg)
    else:
        _migrate_user_config(bundled, user_cfg)
    return user_cfg


def _resolve_media_dir() -> Path:
    """Locate the bundled media/ directory. Nuitka maps '--include-data-dir=media=media'
    to the dist root, which is also Path(__file__).parent for this module once compiled;
    in an uncompiled run media/ instead sits one level above src/."""
    return (Path(__file__).parent if IS_FROZEN else Path(__file__).parent.parent) / 'media'


def _register_overlay_font() -> None:
    """Register the bundled JetBrains Mono cuts with Qt's font database so the
    intravascular metrics overlay (the only place that uses them, see metrics.py) can
    address them by family name. The app-wide font stays the PyQt6/Windows default.

    A missing or unreadable font is not fatal - the overlay falls back to the system
    font if the family can't be resolved at draw time."""
    font_dir = _resolve_media_dir() / 'fonts'
    for ttf in sorted(font_dir.glob('JetBrainsMono-*.ttf')):
        if QFontDatabase.addApplicationFont(str(ttf)) == -1:
            logger.warning(f'Could not load bundled font {ttf.name}; skipping it')


def main() -> None:
    config = _load_config(_resolve_config_path())
    app = QApplication(sys.argv)
    app.setApplicationVersion(__version__)

    # Before Master(): the intravascular metrics overlay addresses this font by family
    # name (see metrics.py) and needs it registered before any frame is drawn.
    _register_overlay_font()
    qdarktheme.setup_theme('dark')  # switch to auto to recognize system mode
    _window = Master(config)

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
