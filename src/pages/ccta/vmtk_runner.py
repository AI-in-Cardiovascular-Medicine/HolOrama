"""Drives vmtk (an external tool, installed separately by the user — never bundled
with this app) to compute aortic-root/RCA/LCA centerlines from a cut-and-smoothed
CCTA surface.

This particular vmtk install is a WSL-native Linux build: its Python venv symlinks
straight to /usr/bin/python3.10, and its vmtkcenterlines/etc. "executables" are
Python scripts with a #!/usr/bin/env python shebang — none of which native Windows
can run directly. So every vmtk invocation here shells out to `wsl.exe`, sourcing
the same two environment scripts the original bash pipeline did (the venv's
bin/activate, then the vmtk build's vmtk_env.sh) before running each command. Every
subprocess call is still driven from Python (subprocess.Popen) — WSL is just the
final process launched, exactly as it would be if the user ran it by hand.

Re-implements, in Python, the three vmtk CLI steps that pipeline previously ran by
hand for each of ao/rca/lca: vmtkcenterlines -> vmtkcenterlinesmoothing ->
vmtksurfacewriter (binary -> ascii).
"""

import os
import threading
import time
from pathlib import Path
from subprocess import PIPE, STDOUT, Popen

import numpy as np
from loguru import logger

_REQUIRED_EXES = ('vmtkcenterlines', 'vmtkcenterlinesmoothing', 'vmtksurfacewriter')
_HEARTBEAT_SECONDS = 15  # how often to print "still running" while a step is silent


def _to_wsl_path(win_path: str) -> str:
    """Convert a Windows path (e.g. D:\\foo\\bar) to its WSL equivalent (/mnt/d/foo/bar).
    WSL mounts every Windows drive under /mnt/<lowercase drive letter>/."""
    p = Path(win_path)
    drive = p.drive.rstrip(':').lower()
    rest = str(p)[len(p.drive) :].replace('\\', '/').lstrip('/')
    return f'/mnt/{drive}/{rest}' if drive else str(p).replace('\\', '/')


def _activation_preamble(venv_path: str, build_path: str) -> str:
    """The same two `source` steps the original bash pipeline ran: the plain Python
    venv, then the vmtk build's own env script (PATH/PYTHONPATH to its bin/lib)."""
    venv_activate = _to_wsl_path(str(Path(venv_path) / 'bin' / 'activate'))
    vmtk_env_sh = _to_wsl_path(str(Path(build_path) / 'vmtk_env.sh'))
    return f'source "{venv_activate}" && source "{vmtk_env_sh}"'


def _run_wsl(command: str, distro: str = '', log_cb=None) -> tuple[int, str]:
    """Run `command` inside WSL, streaming its combined stdout+stderr to log_cb line
    by line AS IT RUNS (not just after it finishes) — vmtk's centerline extraction
    can run silently for a long time, and blocking on .communicate() until the whole
    thing exits gave no way to tell a slow run from a frozen one. Also runs a
    heartbeat print every _HEARTBEAT_SECONDS while the step is otherwise silent, for
    the same reason. Returns (exit_code, full_combined_output).
    """
    # A machine can have several WSL distros installed side by side (e.g. a fresh
    # default "Ubuntu-22.04" alongside an older "Ubuntu" that actually has vmtk's
    # dependencies) — `wsl.exe` with no -d targets whichever one is set as default,
    # which is not necessarily the one vmtk was built/tested against. Pass distro
    # explicitly (config.yaml: vmtk.wsl_distro) whenever it's known.
    cmd = ['wsl'] + (['-d', distro] if distro else []) + ['bash', '-lc', command]
    try:
        process = Popen(cmd, stdout=PIPE, stderr=STDOUT, text=True, bufsize=1)
    except FileNotFoundError as e:
        raise RuntimeError('wsl.exe not found — WSL does not appear to be installed on this machine.') from e

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
                log_cb(line)
    finally:
        stop_heartbeat.set()
        hb_thread.join(timeout=1.0)

    process.wait()
    return process.returncode, '\n'.join(lines)


def check_vmtk_available(venv_path: str, build_path: str, distro: str = '') -> tuple[bool, str]:
    """Verify vmtk can actually be started: both activation scripts exist on disk,
    WSL is available, and the 3 required executables resolve on PATH after sourcing
    them — in the given WSL distro specifically (see _run_wsl). Returns (True, '')
    on success, or (False, reason) with a specific, actionable explanation — this is
    the "can vmtk actually be started" gate the Calculate Centerlines button checks
    before doing anything else.
    """
    if not venv_path or not build_path:
        return False, 'vmtk.venv_path and/or vmtk.build_path are empty in config.yaml.'

    venv_activate = Path(venv_path) / 'bin' / 'activate'
    if not venv_activate.is_file():
        return False, f'venv activate script not found: {venv_activate}'

    vmtk_env_sh = Path(build_path) / 'vmtk_env.sh'
    if not vmtk_env_sh.is_file():
        return False, f'vmtk build env script not found: {vmtk_env_sh}'

    preamble = _activation_preamble(venv_path, build_path)
    check_cmd = preamble + ''.join(f' && command -v {name} > /dev/null' for name in _REQUIRED_EXES)
    try:
        code, output = _run_wsl(check_cmd, distro)
    except RuntimeError as e:
        return False, str(e)

    if code != 0:
        distro_note = f' in WSL distro "{distro}"' if distro else ' in the default WSL distro'
        return False, (
            f'vmtk executables not resolvable{distro_note} after activating both environments '
            f'(exit {code}). {output.strip()}'
        )
    return True, ''


def write_point_csv(path: str, points: list[np.ndarray]) -> None:
    """Write points in the exact column layout the original vmtk pipeline expects:
    header `Point ID,Points_0,Points_1,Points_2,Points_Magnitude`, one row per point.
    Point ID and magnitude aren't read by vmtk itself (only Points_0..2 are), but are
    included for format fidelity with the rest of the pipeline's file layout. Written
    directly by Python on the Windows side — WSL sees the same file via /mnt/<drive>.
    """
    with open(path, 'w', newline='') as f:
        f.write('Point ID,Points_0,Points_1,Points_2,Points_Magnitude\n')
        for i, p in enumerate(points):
            x, y, z = float(p[0]), float(p[1]), float(p[2])
            magnitude = float(np.linalg.norm(p))
            f.write(f'{i},{x:.6f},{y:.6f},{z:.6f},{magnitude:.6f}\n')


def _fmt_point(p: np.ndarray) -> str:
    return f'{float(p[0]):.6f} {float(p[1]):.6f} {float(p[2]):.6f}'


def _run_step(preamble: str, exe: str, args: list[str], distro: str, log_cb=None) -> None:
    """Runs one vmtk CLI step, printing clear start/finish markers regardless of
    whether the tool itself prints anything — vmtkcenterlines in particular can run
    for a long time with zero output during its Voronoi-diagram computation, which
    otherwise looks identical to the process having frozen."""
    command = f'{preamble} && {exe} ' + ' '.join(args)
    if log_cb is not None:
        log_cb(f'>>> starting {exe} ...')
    start = time.monotonic()
    code, output = _run_wsl(command, distro, log_cb)
    elapsed = time.monotonic() - start
    if code != 0:
        raise RuntimeError(f'{exe} failed after {elapsed:.1f}s (exit {code}): {output.strip()[-2000:]}')
    if log_cb is not None:
        log_cb(f'<<< {exe} finished in {elapsed:.1f}s')


def _run_centerline(
    label: str,
    preamble: str,
    stl_path: str,
    out_dir: str,
    source: np.ndarray,
    targets: list[np.ndarray],
    distro: str,
    log_cb=None,
) -> str:
    """Windows-side paths in, Windows-side final path out — the WSL path translation
    happens only for the strings actually embedded in each vmtk command."""
    raw_out = os.path.join(out_dir, f'{label}_cl_raw.vtp')
    final_out = os.path.join(out_dir, f'{label}_cl.vtp')
    stl_wsl = _to_wsl_path(stl_path)
    raw_wsl = _to_wsl_path(raw_out)
    final_wsl = _to_wsl_path(final_out)
    target_args = ' '.join(_fmt_point(p) for p in targets)

    if log_cb is not None:
        log_cb(f'=== Computing centerline: {label} ===')

    _run_step(
        preamble,
        'vmtkcenterlines',
        [
            '-seedselector',
            'pointlist',
            '-ifile',
            f'"{stl_wsl}"',
            '-sourcepoints',
            _fmt_point(source),
            '-targetpoints',
            target_args,
            '-endpoints',
            '1',
            '-ofile',
            f'"{raw_wsl}"',
        ],
        distro,
        log_cb,
    )
    _run_step(
        preamble,
        'vmtkcenterlinesmoothing',
        ['-ifile', f'"{raw_wsl}"', '-iterations', '300', '-ofile', f'"{raw_wsl}"'],
        distro,
        log_cb,
    )
    _run_step(
        preamble,
        'vmtksurfacewriter',
        ['-ifile', f'"{raw_wsl}"', '-mode', 'ascii', '-ofile', f'"{final_wsl}"'],
        distro,
        log_cb,
    )

    try:
        os.remove(raw_out)
    except OSError:
        logger.warning(f'Could not remove intermediate centerline file: {raw_out}')

    return final_out


def run_centerlines(
    stl_path: str,
    out_dir: str,
    ao_source: np.ndarray,
    ao_target: np.ndarray,
    rca_targets: list[np.ndarray],
    lca_targets: list[np.ndarray],
    venv_path: str,
    build_path: str,
    distro: str = '',
    log_cb=None,
) -> dict[str, str]:
    """Compute the aortic-root, RCA, and LCA centerlines, writing `<label>_cl.vtp`
    into out_dir (a Windows path) for each. ao_source is the shared source point
    (aortic root/LV side) for all three; ao_target/rca_targets/lca_targets are the
    per-centerline target points."""
    preamble = _activation_preamble(venv_path, build_path)
    return {
        'ao': _run_centerline('ao', preamble, stl_path, out_dir, ao_source, [ao_target], distro, log_cb),
        'rca': _run_centerline('rca', preamble, stl_path, out_dir, ao_source, rca_targets, distro, log_cb),
        'lca': _run_centerline('lca', preamble, stl_path, out_dir, ao_source, lca_targets, distro, log_cb),
    }
