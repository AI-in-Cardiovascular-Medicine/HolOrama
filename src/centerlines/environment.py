"""The pinned conda-forge environment vmtk runs in, managed by pixi (https://pixi.sh).

vmtk only ships as a conda-forge package (no pip wheels), so it cannot live in HolOrama's own
Python environment or be compiled into the installer. Instead:

- vmtk_env/pixi.toml + pixi.lock (next to this module, bundled with the app) pin the
  environment exactly.
- On first use, setup_env() copies both into <user data dir>/vmtk-env (or
  config.centerlines.env_dir), downloads a pinned pixi binary if none is available, and runs
  `pixi install --frozen` there. A marker file records which manifest/lock was installed, so an
  app update that changes the pin re-syncs the environment automatically.
"""

from __future__ import annotations

import hashlib
import io
import os
import platform
import shutil
import stat
import tarfile
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path

from app_paths import user_data_dir
from centerlines.process import run_streaming
from domain.ccta.centerline_types import VmtkEnv

BUNDLED_ENV_DIR = Path(__file__).resolve().parent / 'vmtk_env'
_ENV_FILES = ('pixi.toml', 'pixi.lock')
_MARKER = '.holorama-env'

# Pinned so a pixi release can never change how the environment is installed under us.
PIXI_VERSION = '0.84.1'
_PIXI_ASSETS = {
    ('Windows', 'AMD64'): 'pixi-x86_64-pc-windows-msvc.zip',
    ('Linux', 'x86_64'): 'pixi-x86_64-unknown-linux-musl.tar.gz',
    ('Darwin', 'x86_64'): 'pixi-x86_64-apple-darwin.tar.gz',
}


def env_from_config(config) -> VmtkEnv:
    """config.yaml's centerlines section. An empty env_dir means <user data dir>/vmtk-env."""
    section = getattr(config, 'centerlines', None)
    env_dir = getattr(section, 'env_dir', '') or ''
    pixi_path = getattr(section, 'pixi_path', '') or ''
    return VmtkEnv(Path(env_dir) if env_dir else user_data_dir() / 'vmtk-env', pixi_path)


def env_ready(env: VmtkEnv) -> tuple[bool, str]:
    """(True, '') if Calculate Centerlines can run right away, else (False, reason). A
    missing or outdated environment is not an error, setup_env() fixes it."""
    ok, reason = platform_supported()
    if not ok:
        return False, reason
    if find_pixi(env) is None:
        if env.pixi_path:
            return False, f'centerlines.pixi_path in config.yaml does not exist: {env.pixi_path}'
        return False, 'The vmtk environment has not been set up yet.'
    marker = env.env_dir / _MARKER
    if not marker.is_file():
        return False, 'The vmtk environment has not been set up yet.'
    if marker.read_text(encoding='utf-8').strip() != _bundled_hash():
        return False, 'The vmtk environment is outdated (this HolOrama version pins a different vmtk).'
    return True, ''


def setup_env(env: VmtkEnv, log_cb: Callable[[str], None] = print) -> None:
    """Create or re-sync the vmtk environment. Downloads and unpacks ~1.5 GB of conda packages the
    first time and takes a few minutes, streaming progress to log_cb. Raises RuntimeError on failure."""
    ok, reason = platform_supported()
    if not ok:
        raise RuntimeError(reason)
    pixi = find_pixi(env)
    if pixi is None:
        if env.pixi_path:
            raise RuntimeError(f'centerlines.pixi_path in config.yaml does not exist: {env.pixi_path}')
        pixi = _download_pixi(log_cb)

    env.env_dir.mkdir(parents=True, exist_ok=True)
    (env.env_dir / _MARKER).unlink(missing_ok=True)
    for name in _ENV_FILES:
        shutil.copyfile(BUNDLED_ENV_DIR / name, env.env_dir / name)

    log_cb(f'Installing the vmtk environment into {env.env_dir} (first time: several minutes) ...')
    code, output = run_streaming([pixi, 'install', '--frozen', '--manifest-path', str(env.manifest)], log_cb)
    if code != 0:
        raise RuntimeError(f'pixi install failed (exit {code}): {output.strip()[-2000:]}')
    (env.env_dir / _MARKER).write_text(_bundled_hash(), encoding='utf-8')
    log_cb('vmtk environment ready.')


def platform_supported() -> tuple[bool, str]:
    key = (platform.system(), platform.machine())
    if key not in _PIXI_ASSETS:
        return False, (
            f'Calculate Centerlines is not available on {key[0]} {key[1]}: vmtk has no conda-forge build for it '
            '(supported: Windows x64, Linux x64, macOS Intel).'
        )
    return True, ''


def find_pixi(env: VmtkEnv) -> str | None:
    """The pixi to use: config.centerlines.pixi_path, else pixi on PATH, else the copy
    setup_env() downloaded. None if there is none yet."""
    if env.pixi_path:
        return env.pixi_path if Path(env.pixi_path).is_file() else None
    on_path = shutil.which('pixi')
    if on_path:
        return on_path
    downloaded = downloaded_pixi_path()
    return str(downloaded) if downloaded.is_file() else None


def downloaded_pixi_path() -> Path:
    return user_data_dir() / 'pixi' / ('pixi.exe' if os.name == 'nt' else 'pixi')


def _bundled_hash() -> str:
    digest = hashlib.sha256()
    for name in _ENV_FILES:
        digest.update((BUNDLED_ENV_DIR / name).read_bytes())
    return digest.hexdigest()


def _download_pixi(log_cb: Callable[[str], None]) -> str:
    asset = _PIXI_ASSETS[(platform.system(), platform.machine())]
    url = f'https://github.com/prefix-dev/pixi/releases/download/v{PIXI_VERSION}/{asset}'
    log_cb(f'Downloading pixi {PIXI_VERSION} ...')
    with urllib.request.urlopen(url, timeout=120) as response:
        data = response.read()
    target = downloaded_pixi_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    if asset.endswith('.zip'):
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            target.write_bytes(zf.read(target.name))
    else:
        with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as tf:
            member = next(m for m in tf.getmembers() if Path(m.name).name == target.name)
            extracted = tf.extractfile(member)
            assert extracted is not None
            target.write_bytes(extracted.read())
    target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    log_cb(f'pixi installed to {target}')
    return str(target)
