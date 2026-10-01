"""Where the app keeps what it writes on its own (logs, its config file, user presets).

When frozen (Nuitka standalone), the app may be installed under a read-only location such
as C:\\Program Files, and the shortcut's working directory points there. Everything the app
writes on its own must therefore live in a per-user, always-writable directory instead of
next to the exe / relative to the CWD (which raises PermissionError on startup). User data
(contours, reports, NIfTi/STL exports) is unaffected: it keeps writing next to the opened
data file. Uncompiled dev runs keep the original in-repo paths.
"""

import os
from pathlib import Path

IS_FROZEN = "__compiled__" in globals()

# The repository root in a dev run (this module sits in src/).
REPO_ROOT = Path(__file__).resolve().parent.parent


def user_data_dir() -> Path:
    base = Path(os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA") or Path.home())
    return base / "HolOrama"
