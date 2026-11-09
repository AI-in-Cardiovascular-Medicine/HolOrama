"""Drives vmtk to compute aortic-root/RCA/LCA centerlines from a cut-and-smoothed CCTA surface.

vmtk only ships as a conda-forge package (no pip wheels), so it cannot live in HolOrama's own
Python environment or be compiled into the installer. Instead it runs in a separate, pinned
conda-forge environment managed by pixi (https://pixi.sh):

- vmtk_env/pixi.toml + pixi.lock (next to this module, bundled with the app) pin the
  environment exactly.
- On first use, setup_env() copies both into <user data dir>/vmtk-env (or
  config.centerlines.env_dir), downloads a pinned pixi binary if none is available, and runs
  `pixi install --frozen` there. A marker file records which manifest/lock was installed, so an
  app update that changes the pin re-syncs the environment automatically.
- Every vmtk step then runs as `pixi run python -m vmtk.<script> ...`: a plain subprocess with
  an argument list (no shell, no WSL, no activation scripts), streaming its output live.

The three CLI steps per centerline are unchanged from the original bash pipeline:
vmtkcenterlines -> vmtkcenterlinesmoothing -> vmtksurfacewriter (binary -> ascii).
"""

from __future__ import annotations

import hashlib
import io
import os
import platform
import shutil
import stat
import subprocess
import tarfile
import threading
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger

from app_paths import user_data_dir

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
_HEARTBEAT_SECONDS = 15  # how often to print "still running" while a step is silent
# No console window flashing up for every step when HolOrama runs as a windowed app.
_NO_WINDOW = getattr(subprocess, 'CREATE_NO_WINDOW', 0)


@dataclass
class SmoothingParams:
    """vmtkcenterlinesmoothing's only two parameters — a moving-average filter over
    the centerline points. Higher iterations/factor = more aggressive smoothing."""

    iterations: int = 100
    factor: float = 0.1


@dataclass(frozen=True)
class VmtkEnv:
    """Where the vmtk environment lives and which pixi manages it."""

    env_dir: Path
    pixi_path: str = ''  # explicit pixi executable from config.yaml; '' = auto

    @classmethod
    def from_config(cls, config) -> VmtkEnv:
        section = getattr(config, 'centerlines', None)
        env_dir = getattr(section, 'env_dir', '') or ''
        pixi_path = getattr(section, 'pixi_path', '') or ''
        return cls(Path(env_dir) if env_dir else user_data_dir() / 'vmtk-env', pixi_path)

    @property
    def manifest(self) -> Path:
        return self.env_dir / 'pixi.toml'

    @property
    def downloaded_pixi(self) -> Path:
        return user_data_dir() / 'pixi' / ('pixi.exe' if os.name == 'nt' else 'pixi')


# -- environment management --------------------------------------------------------------
def _bundled_hash() -> str:
    digest = hashlib.sha256()
    for name in _ENV_FILES:
        digest.update((BUNDLED_ENV_DIR / name).read_bytes())
    return digest.hexdigest()


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
    return str(env.downloaded_pixi) if env.downloaded_pixi.is_file() else None


def env_ready(env: VmtkEnv) -> tuple[bool, str]:
    """(True, '') if Calculate Centerlines can run right away, else (False, reason). A
    missing or outdated environment is not an error — setup_env() fixes it."""
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


def _download_pixi(env: VmtkEnv, log_cb=print) -> str:
    asset = _PIXI_ASSETS[(platform.system(), platform.machine())]
    url = f'https://github.com/prefix-dev/pixi/releases/download/v{PIXI_VERSION}/{asset}'
    log_cb(f'Downloading pixi {PIXI_VERSION} ...')
    with urllib.request.urlopen(url, timeout=120) as response:
        data = response.read()
    target = env.downloaded_pixi
    target.parent.mkdir(parents=True, exist_ok=True)
    name = target.name
    if asset.endswith('.zip'):
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            target.write_bytes(zf.read(name))
    else:
        with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as tf:
            member = next(m for m in tf.getmembers() if Path(m.name).name == name)
            extracted = tf.extractfile(member)
            assert extracted is not None
            target.write_bytes(extracted.read())
    target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    log_cb(f'pixi installed to {target}')
    return str(target)


def setup_env(env: VmtkEnv, log_cb=print) -> None:
    """Create or re-sync the vmtk environment. Downloads and unpacks ~1.5 GB of conda packages the first
    time and takes a few minutes; streams progress to log_cb. Raises RuntimeError on failure."""
    ok, reason = platform_supported()
    if not ok:
        raise RuntimeError(reason)
    pixi = find_pixi(env)
    if pixi is None:
        if env.pixi_path:
            raise RuntimeError(f'centerlines.pixi_path in config.yaml does not exist: {env.pixi_path}')
        pixi = _download_pixi(env, log_cb)

    env.env_dir.mkdir(parents=True, exist_ok=True)
    (env.env_dir / _MARKER).unlink(missing_ok=True)
    for name in _ENV_FILES:
        shutil.copyfile(BUNDLED_ENV_DIR / name, env.env_dir / name)

    log_cb(f'Installing the vmtk environment into {env.env_dir} (first time: several minutes) ...')
    code, output = _run([pixi, 'install', '--frozen', '--manifest-path', str(env.manifest)], log_cb)
    if code != 0:
        raise RuntimeError(f'pixi install failed (exit {code}): {output.strip()[-2000:]}')
    (env.env_dir / _MARKER).write_text(_bundled_hash(), encoding='utf-8')
    log_cb('vmtk environment ready.')


# -- running vmtk ------------------------------------------------------------------------
def _run(cmd: list[str], log_cb=None) -> tuple[int, str]:
    """Run cmd, streaming its combined stdout+stderr to log_cb line by line AS IT RUNS —
    vmtk's centerline extraction can run silently for a long time, so a heartbeat line is
    also printed every _HEARTBEAT_SECONDS of silence to tell slow from frozen.
    Returns (exit_code, full_combined_output)."""
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

    start = time.monotonic()
    last_output = [start]  # 1-elem list so the heartbeat thread can read/reset it without `nonlocal`
    stop_heartbeat = threading.Event()

    def _heartbeat() -> None:
        while not stop_heartbeat.wait(1.0):
            if log_cb is not None and time.monotonic() - last_output[0] >= _HEARTBEAT_SECONDS:
                log_cb(f'    ...still running ({int(time.monotonic() - start)}s elapsed, no output yet)')
                last_output[0] = time.monotonic()

    hb_thread = threading.Thread(target=_heartbeat, daemon=True)
    hb_thread.start()

    lines: list[str] = []
    assert process.stdout is not None
    try:
        for line in process.stdout:
            line = line.rstrip('\n')
            lines.append(line)
            last_output[0] = time.monotonic()
            if log_cb is not None:
                # pixi prints symbols such as a check mark; a Windows console (cp1252) cannot
                # encode them and print() would raise mid-run.
                log_cb(line.encode('ascii', 'replace').decode('ascii'))
    finally:
        stop_heartbeat.set()
        hb_thread.join(timeout=1.0)

    process.wait()
    return process.returncode, '\n'.join(lines)


def _run_step(env: VmtkEnv, pixi: str, script: str, args: list[str], log_cb=None) -> None:
    """Runs one vmtk script inside the environment, printing clear start/finish markers
    regardless of whether the tool itself prints anything."""
    cmd = [pixi, 'run', '--frozen', '--manifest-path', str(env.manifest), 'python', '-m', f'vmtk.{script}', *args]
    if log_cb is not None:
        log_cb(f'>>> starting {script} ...')
    start = time.monotonic()
    code, output = _run(cmd, log_cb)
    elapsed = time.monotonic() - start
    if code != 0:
        raise RuntimeError(f'{script} failed after {elapsed:.1f}s (exit {code}): {output.strip()[-2000:]}')
    if log_cb is not None:
        log_cb(f'<<< {script} finished in {elapsed:.1f}s')


def write_point_csv(path: str, points: list[np.ndarray]) -> None:
    """Write points in the exact column layout the original vmtk pipeline expects:
    header `Point ID,Points_0,Points_1,Points_2,Points_Magnitude`, one row per point.
    Point ID and magnitude aren't read by vmtk itself (only Points_0..2 are), but are
    included for format fidelity with the rest of the pipeline's file layout."""
    with open(path, 'w', newline='') as f:
        f.write('Point ID,Points_0,Points_1,Points_2,Points_Magnitude\n')
        for i, p in enumerate(points):
            x, y, z = float(p[0]), float(p[1]), float(p[2])
            magnitude = float(np.linalg.norm(p))
            f.write(f'{i},{x:.6f},{y:.6f},{z:.6f},{magnitude:.6f}\n')


def _point_args(points: list[np.ndarray]) -> list[str]:
    return [f'{float(c):.6f}' for p in points for c in p[:3]]


def _path_arg(path: str) -> str:
    """vmtk's argument parser (pypescript.ParseArguments) exec()s every value as a Python
    string literal, so the backslashes of a Windows path are read as escapes (C:\\Users ->
    a broken \\U escape). Forward slashes work everywhere; a quote cannot be passed at all."""
    if "'" in path:
        raise RuntimeError(f"vmtk cannot handle a ' in file paths: {path}")
    return path.replace('\\', '/')


def _run_centerline(
    env: VmtkEnv,
    pixi: str,
    label: str,
    stl_path: str,
    out_dir: str,
    source: np.ndarray,
    targets: list[np.ndarray],
    smoothing: SmoothingParams,
    log_cb=None,
) -> str:
    raw_out = os.path.join(out_dir, f'{label}_cl_raw.vtp')
    final_out = os.path.join(out_dir, f'{label}_cl.vtp')
    stl_arg, raw_arg, final_arg = _path_arg(stl_path), _path_arg(raw_out), _path_arg(final_out)

    if log_cb is not None:
        log_cb(f'=== Computing centerline: {label} ===')

    _run_step(
        env,
        pixi,
        'vmtkcenterlines',
        ['-seedselector', 'pointlist', '-ifile', stl_arg]
        + ['-sourcepoints', *_point_args([source])]
        + ['-targetpoints', *_point_args(targets)]
        + ['-endpoints', '1', '-ofile', raw_arg],
        log_cb,
    )
    _run_step(
        env,
        pixi,
        'vmtkcenterlinesmoothing',
        ['-ifile', raw_arg, '-iterations', str(smoothing.iterations), '-factor', str(smoothing.factor)]
        + ['-ofile', raw_arg],
        log_cb,
    )
    _run_step(env, pixi, 'vmtksurfacewriter', ['-ifile', raw_arg, '-mode', 'ascii', '-ofile', final_arg], log_cb)

    try:
        os.remove(raw_out)
    except OSError:
        logger.warning(f'Could not remove intermediate centerline file: {raw_out}')

    return final_out


def run_centerlines(
    env: VmtkEnv,
    stl_path: str,
    out_dir: str,
    ao_source: np.ndarray,
    ao_target: np.ndarray,
    rca_targets: list[np.ndarray],
    lca_targets: list[np.ndarray],
    ao_smoothing: SmoothingParams,
    coronary_smoothing: SmoothingParams,
    log_cb=None,
) -> dict[str, str]:
    """Compute the aortic-root, RCA, and LCA centerlines, writing `<label>_cl.vtp`
    into out_dir for each. ao_source is the shared source point (aorta-top cut plane)
    for all three; ao_target/rca_targets/lca_targets are the per-centerline target
    points. ao_smoothing/coronary_smoothing are separate since the aorta often needs
    much heavier smoothing than the finer coronary branches. The environment must be
    ready (env_ready); raises RuntimeError otherwise or if a step fails."""
    ready, reason = env_ready(env)
    if not ready:
        raise RuntimeError(reason)
    pixi = find_pixi(env)
    assert pixi is not None
    return {
        'ao': _run_centerline(env, pixi, 'ao', stl_path, out_dir, ao_source, [ao_target], ao_smoothing, log_cb),
        'rca': _run_centerline(env, pixi, 'rca', stl_path, out_dir, ao_source, rca_targets, coronary_smoothing, log_cb),
        'lca': _run_centerline(env, pixi, 'lca', stl_path, out_dir, ao_source, lca_targets, coronary_smoothing, log_cb),
    }
