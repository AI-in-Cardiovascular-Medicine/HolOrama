"""Compute the aortic-root, RCA and LCA centerlines of a cut CCTA surface with vmtk.

Every vmtk step runs as `pixi run python -m vmtk.<script> ...` in the environment of
environment.py: a plain subprocess with an argument list (no shell, no WSL, no activation
scripts), streaming its output live. The three steps per centerline are unchanged from the
original bash pipeline: vmtkcenterlines -> vmtkcenterlinesmoothing -> vmtksurfacewriter
(binary -> ascii).
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable

import numpy as np
from loguru import logger

from centerlines import environment
from centerlines.process import run_streaming
from domain.ccta.centerline_types import CenterlineInputs, SmoothingParams, VmtkEnv

LogCallback = Callable[[str], None] | None


def compute_centerlines(
    env: VmtkEnv,
    inputs: CenterlineInputs,
    ao_smoothing: SmoothingParams,
    coronary_smoothing: SmoothingParams,
    log_cb: LogCallback = None,
) -> dict[str, str]:
    """Export the cut mesh and the seed points into inputs.out_dir, then run vmtk on them.
    Returns {'ao' | 'rca' | 'lca': path of the centerline .vtp}."""
    stl_path = os.path.join(inputs.out_dir, f'{inputs.stem}_root_smooth.stl')
    inputs.cut_mesh.export(stl_path)
    write_point_csv(os.path.join(inputs.out_dir, 'ao.csv'), [inputs.inlet, inputs.outlet])
    write_point_csv(os.path.join(inputs.out_dir, 'rca.csv'), inputs.rca_points)
    write_point_csv(os.path.join(inputs.out_dir, 'lca.csv'), inputs.lca_points)
    return run_centerlines(
        env,
        stl_path,
        inputs.out_dir,
        ao_source=inputs.inlet,
        ao_target=inputs.outlet,
        rca_targets=inputs.rca_points,
        lca_targets=inputs.lca_points,
        ao_smoothing=ao_smoothing,
        coronary_smoothing=coronary_smoothing,
        log_cb=log_cb,
    )


def write_point_csv(path: str, points: list[np.ndarray]) -> None:
    """Write points in the column layout of the original vmtk pipeline:
    header `Point ID,Points_0,Points_1,Points_2,Points_Magnitude`, one row per point.
    vmtk itself only reads Points_0..2, the other columns keep the pipeline's file layout."""
    with open(path, 'w', newline='') as f:
        f.write('Point ID,Points_0,Points_1,Points_2,Points_Magnitude\n')
        for i, p in enumerate(points):
            x, y, z = float(p[0]), float(p[1]), float(p[2])
            magnitude = float(np.linalg.norm(p))
            f.write(f'{i},{x:.6f},{y:.6f},{z:.6f},{magnitude:.6f}\n')


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
    log_cb: LogCallback = None,
) -> dict[str, str]:
    """Compute the aortic-root, RCA and LCA centerlines, writing `<label>_cl.vtp` into out_dir
    for each. ao_source (the aorta-top cut plane) is the shared source point of all three. The
    environment must be ready (environment.env_ready), raises RuntimeError otherwise or if a step fails."""
    ready, reason = environment.env_ready(env)
    if not ready:
        raise RuntimeError(reason)
    pixi = environment.find_pixi(env)
    assert pixi is not None
    return {
        'ao': _run_centerline(env, pixi, 'ao', stl_path, out_dir, ao_source, [ao_target], ao_smoothing, log_cb),
        'rca': _run_centerline(env, pixi, 'rca', stl_path, out_dir, ao_source, rca_targets, coronary_smoothing, log_cb),
        'lca': _run_centerline(env, pixi, 'lca', stl_path, out_dir, ao_source, lca_targets, coronary_smoothing, log_cb),
    }


def _run_centerline(
    env: VmtkEnv,
    pixi: str,
    label: str,
    stl_path: str,
    out_dir: str,
    source: np.ndarray,
    targets: list[np.ndarray],
    smoothing: SmoothingParams,
    log_cb: LogCallback,
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


def _path_arg(path: str) -> str:
    """vmtk's argument parser (pypescript.ParseArguments) exec()s every value as a Python
    string literal, so the backslashes of a Windows path are read as escapes (C:\\Users ->
    a broken \\U escape). Forward slashes work everywhere, a quote cannot be passed at all."""
    if "'" in path:
        raise RuntimeError(f"vmtk cannot handle a ' in file paths: {path}")
    return path.replace('\\', '/')


def _point_args(points: list[np.ndarray]) -> list[str]:
    return [f'{float(c):.6f}' for p in points for c in p[:3]]


def _run_step(env: VmtkEnv, pixi: str, script: str, args: list[str], log_cb: LogCallback) -> None:
    """Run one vmtk script inside the environment, with start and finish markers whether or
    not the script prints anything itself."""
    cmd = [pixi, 'run', '--frozen', '--manifest-path', str(env.manifest), 'python', '-m', f'vmtk.{script}', *args]
    if log_cb is not None:
        log_cb(f'>>> starting {script} ...')
    start = time.monotonic()
    code, output = run_streaming(cmd, log_cb)
    elapsed = time.monotonic() - start
    if code != 0:
        raise RuntimeError(f'{script} failed after {elapsed:.1f}s (exit {code}): {output.strip()[-2000:]}')
    if log_cb is not None:
        log_cb(f'<<< {script} finished in {elapsed:.1f}s')
