"""Calculate Centerlines without Qt (src/centerlines): the pixi-managed vmtk environment and
the vmtk commands run in it.

Nothing here downloads or runs vmtk, subprocess calls are intercepted. The real environment
is exercised manually via Calculate Centerlines (see docs/contents/installation.rst)."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh

from centerlines import environment, process, vmtk
from domain.ccta.centerline_types import (
    AORTA_SMOOTHING,
    CORONARY_SMOOTHING,
    CenterlineInputs,
    SmoothingParams,
    VmtkEnv,
)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(environment, 'user_data_dir', lambda: tmp_path / 'user')
    monkeypatch.setattr(environment, 'platform_supported', lambda: (True, ''))
    monkeypatch.setattr(environment.shutil, 'which', lambda name: None)
    return VmtkEnv(tmp_path / 'env')


def test_bundled_manifest_and_lock_exist():
    assert (environment.BUNDLED_ENV_DIR / 'pixi.toml').is_file()
    lock = (environment.BUNDLED_ENV_DIR / 'pixi.lock').read_text(encoding='utf-8')
    assert 'vmtk' in lock and 'win-64' in lock and 'linux-64' in lock


def test_env_from_config_defaults_and_overrides(tmp_path, monkeypatch):
    monkeypatch.setattr(environment, 'user_data_dir', lambda: tmp_path)
    assert environment.env_from_config(SimpleNamespace()).env_dir == tmp_path / 'vmtk-env'
    blank = SimpleNamespace(centerlines=SimpleNamespace(env_dir='', pixi_path=''))
    assert environment.env_from_config(blank) == VmtkEnv(tmp_path / 'vmtk-env', '')
    custom = SimpleNamespace(centerlines=SimpleNamespace(env_dir=str(tmp_path / 'x'), pixi_path='C:/pixi.exe'))
    assert environment.env_from_config(custom) == VmtkEnv(tmp_path / 'x', 'C:/pixi.exe')


def test_not_ready_until_set_up(env):
    ready, reason = environment.env_ready(env)
    assert not ready and 'not been set up' in reason


def test_setup_copies_manifest_runs_frozen_install_and_marks_ready(env, monkeypatch):
    pixi = _fake_downloaded_pixi()
    run = _FakeRun()
    monkeypatch.setattr(environment, 'run_streaming', run)

    environment.setup_env(env, log_cb=_ignore)

    assert run.calls == [[str(pixi), 'install', '--frozen', '--manifest-path', str(env.manifest)]]
    for name in ('pixi.toml', 'pixi.lock'):
        assert (env.env_dir / name).read_bytes() == (environment.BUNDLED_ENV_DIR / name).read_bytes()
    assert environment.env_ready(env) == (True, '')


def test_changed_pin_makes_env_outdated(env, monkeypatch):
    _set_up(env, monkeypatch)
    monkeypatch.setattr(environment, '_bundled_hash', lambda: 'a-newer-lock')
    ready, reason = environment.env_ready(env)
    assert not ready and 'outdated' in reason


def test_failed_install_stays_not_ready(env, monkeypatch):
    _fake_downloaded_pixi()
    monkeypatch.setattr(environment, 'run_streaming', _FakeRun(code=1, output='solver exploded'))
    with pytest.raises(RuntimeError, match='solver exploded'):
        environment.setup_env(env, log_cb=_ignore)
    assert not environment.env_ready(env)[0]


def test_explicit_pixi_path_must_exist(env, tmp_path):
    missing = VmtkEnv(env.env_dir, str(tmp_path / 'missing-pixi.exe'))
    ready, reason = environment.env_ready(missing)
    assert not ready and 'pixi_path' in reason


def test_run_centerlines_builds_vmtk_commands(env, monkeypatch, tmp_path):
    pixi = _set_up(env, monkeypatch)
    run = _FakeRun()
    monkeypatch.setattr(vmtk, 'run_streaming', run)
    out_dir = tmp_path / 'out'
    out_dir.mkdir()
    paths = vmtk.run_centerlines(
        env,
        str(out_dir / 'case_root_smooth.stl'),
        str(out_dir),
        ao_source=np.array([0.0, 1.0, 2.0]),
        ao_target=np.array([3.0, 4.0, 5.0]),
        rca_targets=[np.array([6.0, 7.0, 8.0])],
        lca_targets=[np.array([9.0, 10.0, 11.0]), np.array([12.0, 13.0, 14.0])],
        ao_smoothing=SmoothingParams(200, 0.2),
        coronary_smoothing=CORONARY_SMOOTHING,
    )

    assert paths == {label: str(out_dir / f'{label}_cl.vtp') for label in ('ao', 'rca', 'lca')}
    assert len(run.calls) == 9  # 3 steps x 3 vessels
    prefix = [str(pixi), 'run', '--frozen', '--manifest-path', str(env.manifest), 'python', '-m']
    assert all(c[: len(prefix)] == prefix for c in run.calls)
    assert [c[len(prefix)] for c in run.calls[:3]] == [
        'vmtk.vmtkcenterlines',
        'vmtk.vmtkcenterlinesmoothing',
        'vmtk.vmtksurfacewriter',
    ]
    lca = run.calls[6]
    targets = lca[lca.index('-targetpoints') + 1 : lca.index('-endpoints')]
    assert targets == ['9.000000', '10.000000', '11.000000', '12.000000', '13.000000', '14.000000']
    ao_smooth = run.calls[1]
    assert ao_smooth[ao_smooth.index('-iterations') + 1] == '200'


def test_compute_centerlines_exports_surface_and_seed_points(env, monkeypatch, tmp_path):
    _set_up(env, monkeypatch)
    monkeypatch.setattr(vmtk, 'run_streaming', _FakeRun())
    inputs = CenterlineInputs(
        cut_mesh=trimesh.creation.box(),
        inlet=np.array([0.0, 0.0, 0.0]),
        outlet=np.array([0.0, 0.0, 1.0]),
        rca_points=[np.array([1.0, 0.0, 0.0])],
        lca_points=[np.array([0.0, 1.0, 0.0])],
        out_dir=str(tmp_path),
        stem='case.nii.gz',
    )
    paths = vmtk.compute_centerlines(env, inputs, AORTA_SMOOTHING, CORONARY_SMOOTHING)
    assert set(paths) == {'ao', 'rca', 'lca'}
    assert (tmp_path / 'case.nii.gz_root_smooth.stl').is_file()
    assert len((tmp_path / 'ao.csv').read_text().splitlines()) == 3  # header, inlet, outlet


def test_run_centerlines_refuses_when_not_ready(env, tmp_path):
    with pytest.raises(RuntimeError, match='not been set up'):
        vmtk.run_centerlines(
            env,
            'x.stl',
            str(tmp_path),
            np.zeros(3),
            np.zeros(3),
            [],
            [],
            AORTA_SMOOTHING,
            CORONARY_SMOOTHING,
        )


def test_paths_are_passed_with_forward_slashes():
    # vmtk exec()s argument values as string literals: C:\Users would be a broken \U escape.
    assert vmtk._path_arg('C:\\Users\\me\\case dir\\x.stl') == 'C:/Users/me/case dir/x.stl'
    with pytest.raises(RuntimeError, match="'"):
        vmtk._path_arg("C:/O'Brien/x.stl")


def test_point_csv_format(tmp_path):
    path = tmp_path / 'p.csv'
    vmtk.write_point_csv(str(path), [np.array([3.0, 4.0, 0.0])])
    assert Path(path).read_text().splitlines() == [
        'Point ID,Points_0,Points_1,Points_2,Points_Magnitude',
        '0,3.000000,4.000000,0.000000,5.000000',
    ]


def test_run_streaming_returns_exit_code_and_output():
    lines: list[str] = []
    code, output = process.run_streaming([sys.executable, '-c', 'print("one"); print("two")'], lines.append)
    assert code == 0 and output.splitlines() == ['one', 'two'] and lines == ['one', 'two']


def _ignore(line: str) -> None:
    pass


def _fake_downloaded_pixi() -> Path:
    pixi = environment.downloaded_pixi_path()
    pixi.parent.mkdir(parents=True)
    pixi.write_bytes(b'')
    return pixi


def _set_up(env: VmtkEnv, monkeypatch) -> Path:
    """A ready environment (fake pixi, fake install), returns the pixi path."""
    pixi = _fake_downloaded_pixi()
    monkeypatch.setattr(environment, 'run_streaming', _FakeRun())
    environment.setup_env(env, log_cb=_ignore)
    return pixi


class _FakeRun:
    """Stands in for process.run_streaming: records each command and returns (code, output)."""

    def __init__(self, code: int = 0, output: str = '') -> None:
        self.code = code
        self.output = output
        self.calls: list[list[str]] = []

    def __call__(self, cmd: list[str], log_cb=None) -> tuple[int, str]:
        self.calls.append(cmd)
        return self.code, self.output
