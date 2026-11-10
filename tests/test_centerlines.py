"""The pixi-managed vmtk environment (src/pages/ccta/left_half/cut_geometry/vmtk.py).

Nothing here downloads or runs vmtk; subprocess calls are intercepted. The real environment
is exercised manually via Calculate Centerlines (see docs/contents/installation.rst)."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pages.ccta.left_half.cut_geometry import vmtk


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(vmtk, 'user_data_dir', lambda: tmp_path / 'user')
    monkeypatch.setattr(vmtk, 'platform_supported', lambda: (True, ''))
    monkeypatch.setattr(vmtk.shutil, 'which', lambda name: None)
    return vmtk.VmtkEnv(tmp_path / 'env')


def test_bundled_manifest_and_lock_exist():
    assert (vmtk.BUNDLED_ENV_DIR / 'pixi.toml').is_file()
    lock = (vmtk.BUNDLED_ENV_DIR / 'pixi.lock').read_text(encoding='utf-8')
    assert 'vmtk' in lock and 'win-64' in lock and 'linux-64' in lock


def test_from_config_defaults_and_overrides(tmp_path, monkeypatch):
    monkeypatch.setattr(vmtk, 'user_data_dir', lambda: tmp_path)
    assert vmtk.VmtkEnv.from_config(SimpleNamespace()).env_dir == tmp_path / 'vmtk-env'
    blank = SimpleNamespace(centerlines=SimpleNamespace(env_dir='', pixi_path=''))
    assert vmtk.VmtkEnv.from_config(blank) == vmtk.VmtkEnv(tmp_path / 'vmtk-env', '')
    custom = SimpleNamespace(centerlines=SimpleNamespace(env_dir=str(tmp_path / 'x'), pixi_path='C:/pixi.exe'))
    assert vmtk.VmtkEnv.from_config(custom) == vmtk.VmtkEnv(tmp_path / 'x', 'C:/pixi.exe')


def test_not_ready_until_set_up(env):
    ready, reason = vmtk.env_ready(env)
    assert not ready and 'not been set up' in reason


def _fake_pixi(env):
    env.downloaded_pixi.parent.mkdir(parents=True)
    env.downloaded_pixi.write_bytes(b'')


def test_setup_copies_manifest_runs_frozen_install_and_marks_ready(env, monkeypatch):
    _fake_pixi(env)
    calls = []
    monkeypatch.setattr(vmtk, '_run', lambda cmd, log_cb=None: (calls.append(cmd), (0, ''))[1])

    vmtk.setup_env(env, log_cb=lambda line: None)

    assert calls == [[str(env.downloaded_pixi), 'install', '--frozen', '--manifest-path', str(env.manifest)]]
    for name in ('pixi.toml', 'pixi.lock'):
        assert (env.env_dir / name).read_bytes() == (vmtk.BUNDLED_ENV_DIR / name).read_bytes()
    assert vmtk.env_ready(env) == (True, '')


def test_changed_pin_makes_env_outdated(env, monkeypatch):
    _fake_pixi(env)
    monkeypatch.setattr(vmtk, '_run', lambda cmd, log_cb=None: (0, ''))
    vmtk.setup_env(env, log_cb=lambda line: None)
    monkeypatch.setattr(vmtk, '_bundled_hash', lambda: 'a-newer-lock')
    ready, reason = vmtk.env_ready(env)
    assert not ready and 'outdated' in reason


def test_failed_install_stays_not_ready(env, monkeypatch):
    _fake_pixi(env)
    monkeypatch.setattr(vmtk, '_run', lambda cmd, log_cb=None: (1, 'solver exploded'))
    with pytest.raises(RuntimeError, match='solver exploded'):
        vmtk.setup_env(env, log_cb=lambda line: None)
    assert not vmtk.env_ready(env)[0]


def test_explicit_pixi_path_must_exist(env, tmp_path):
    missing = vmtk.VmtkEnv(env.env_dir, str(tmp_path / 'missing-pixi.exe'))
    ready, reason = vmtk.env_ready(missing)
    assert not ready and 'pixi_path' in reason


def test_run_centerlines_builds_vmtk_commands(env, monkeypatch, tmp_path):
    _fake_pixi(env)
    monkeypatch.setattr(vmtk, '_run', lambda cmd, log_cb=None: (0, ''))
    vmtk.setup_env(env, log_cb=lambda line: None)

    calls = []
    monkeypatch.setattr(vmtk, '_run', lambda cmd, log_cb=None: (calls.append(cmd), (0, ''))[1])
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
        ao_smoothing=vmtk.SmoothingParams(200, 0.2),
        coronary_smoothing=vmtk.SmoothingParams(),
    )

    assert paths == {label: str(out_dir / f'{label}_cl.vtp') for label in ('ao', 'rca', 'lca')}
    assert len(calls) == 9  # 3 steps x 3 vessels
    prefix = [str(env.downloaded_pixi), 'run', '--frozen', '--manifest-path', str(env.manifest), 'python', '-m']
    assert all(c[: len(prefix)] == prefix for c in calls)
    assert [c[len(prefix)] for c in calls[:3]] == [
        'vmtk.vmtkcenterlines',
        'vmtk.vmtkcenterlinesmoothing',
        'vmtk.vmtksurfacewriter',
    ]
    lca = calls[6]
    targets = lca[lca.index('-targetpoints') + 1 : lca.index('-endpoints')]
    assert targets == ['9.000000', '10.000000', '11.000000', '12.000000', '13.000000', '14.000000']
    ao_smooth = calls[1]
    assert ao_smooth[ao_smooth.index('-iterations') + 1] == '200'


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
            vmtk.SmoothingParams(),
            vmtk.SmoothingParams(),
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
