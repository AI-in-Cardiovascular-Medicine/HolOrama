"""The fusion case sidecar: `{CCTA mesh stem}_fusion_{version}.json`, next to the mesh.

It holds what took user effort and cannot be recomputed: the label maps of every labeling step
(with the lasso edits), branch names, deleted branches, and the chosen alignment. Nothing about
scaling, point removal or stitching, which are quick to redo.

Labels are stored as vertex indices into the mesh label_geometry labelled, not as coordinates:
a few MB instead of tens, and indices survive scaling (it moves vertices, keeps their order).
Deleting a branch or removing points drops vertices, so a vertex that is gone keeps the label
it had when it was last there. The file always describes the whole original mesh.

Same conventions as the CCTA cut state (pages/ccta/left_half/cut_geometry/state_io.py): the
version is in the filename, the newest file is picked by parsing it, writes are atomic.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import shutil
import tempfile
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np
from loguru import logger

from version import version_file_str

STEP_GEOMETRY = 'label_geometry'
STEP_BRANCHES = 'label_branches'
STEP_ANOMALOUS = 'label_anomalous'
STEPS = (STEP_GEOMETRY, STEP_BRANCHES, STEP_ANOMALOUS)
BRANCH_KEYS = ('rca_branches', 'lca_branches')

_FILENAME_RE = re.compile(r'_fusion_(\d+)_(\d+)_(\d+)\.json$')

# One worker: writes land in the order they were asked for, off the UI thread. Not a daemon,
# so a pending write still finishes when the app closes.
_writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix='fusion-state')
_last_hash: dict[str, str] = {}


class VertexIndex:
    """Original vertex index of a vertex position, for the mesh label_geometry labelled."""

    def __init__(self, vertices) -> None:
        self.vertices = np.array(vertices, dtype=np.float64)
        self._rebuild()

    def _rebuild(self) -> None:
        self._lookup = {v: i for i, v in enumerate(map(tuple, self.vertices.tolist()))}

    def __len__(self) -> int:
        return len(self.vertices)

    def indices(self, points) -> list[int]:
        """Original indices of `points` (vertex positions). Unknown points (patch vertices) are left out."""
        lookup = self._lookup
        found = (lookup.get(tuple(p)) for p in points)
        return [i for i in found if i is not None]

    def present(self, mesh) -> np.ndarray:
        """Mask over the original vertices: which of them `mesh` still has."""
        mask = np.zeros(len(self), dtype=bool)
        mask[self.indices(np.asarray(mesh.vertices).tolist())] = True
        return mask

    def points(self, indices, present: np.ndarray | None = None) -> list[tuple]:
        """Current positions of original vertices, as label lists hold them. With `present`, gone vertices are skipped."""
        idx = np.asarray([i for i in indices if 0 <= i < len(self)], dtype=np.int64)
        if present is not None:
            idx = idx[present[idx]]
        return [tuple(v) for v in self.vertices[idx].tolist()]

    def follow(self, old_mesh, new_mesh) -> None:
        """`new_mesh` is `old_mesh` with vertices moved (scaling: same count and order)."""
        old = np.asarray(old_mesh.vertices, dtype=np.float64)
        new = np.asarray(new_mesh.vertices, dtype=np.float64)
        if old.shape != new.shape:
            raise ValueError('follow() needs meshes with the same vertices, only moved')
        lookup = self._lookup
        rows = [(i, lookup.get(tuple(v))) for i, v in enumerate(old.tolist())]
        known = [(i, o) for i, o in rows if o is not None]
        if known:
            at, origin = map(list, zip(*known))
            self.vertices[origin] = new[at]
        self._rebuild()


def _merge(current: list[int], previous: list[int], present: np.ndarray) -> list[int]:
    """Current members, plus previous members whose vertex is gone (they keep their last label)."""
    kept = [i for i in previous if 0 <= i < len(present) and not present[i]]
    return sorted(set(current).union(kept))


def step_labels(results: dict, keys, index: VertexIndex, previous: dict, present: np.ndarray) -> dict:
    """The label map of one step: key -> original indices, or for branch keys key -> {branch id: indices}.
    JSON keys are strings, so branch ids are too."""
    out: dict = {}
    for key in keys:
        value = results.get(key)
        before = previous.get(key)
        if isinstance(value, dict) or isinstance(before, dict):
            value, before = value or {}, before or {}
            branches = {str(b): index.indices(pts) for b, pts in value.items()}
            for b in set(branches) | set(before):
                merged = _merge(branches.get(b, []), before.get(b, []), present)
                if merged:
                    out.setdefault(key, {})[b] = merged
            out.setdefault(key, {})
        else:
            out[key] = _merge(index.indices(value or []), before or [], present)
    return out


def apply_step_labels(results: dict, labels: dict, index: VertexIndex, present: np.ndarray) -> None:
    """Overwrite the labels of one step in `results` with a saved map, in place. Vertices the mesh no longer
    has are skipped, empty branches dropped."""
    for key, value in labels.items():
        if isinstance(value, dict):
            branches = {int(b): index.points(idx, present) for b, idx in value.items()}
            results[key] = {b: pts for b, pts in sorted(branches.items()) if pts}
        else:
            results[key] = index.points(value, present)


def state_path(mesh_path: str) -> str:
    return f'{os.path.splitext(mesh_path)[0]}_fusion_{version_file_str}.json'


def _sort_key(path: str) -> tuple[int, int, int]:
    match = _FILENAME_RE.search(os.path.basename(path))
    return tuple(int(g) for g in match.groups()) if match else (0, 0, 0)  # type: ignore[return-value]


def find_state(mesh_path: str) -> str | None:
    """The newest state file of this mesh, by the version in its name (mtime does not survive a copy)."""
    matches = glob.glob(f'{glob.escape(os.path.splitext(mesh_path)[0])}_fusion_*.json')
    return max(matches, key=_sort_key) if matches else None


def load_state(path: str) -> dict | None:
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        logger.exception(f'Failed to read fusion state: {path}')
        return None


def write_state(state: dict, path: str, *, blocking: bool = False) -> Future | None:
    """Serialize and write on the writer thread (or here with `blocking`). Skips the write when the content
    did not change. `state` must not be mutated afterwards: build a fresh dict per call."""
    if blocking:
        _write(state, path)
        return None
    return _writer.submit(_write, state, path)


def _write(state: dict, path: str) -> None:
    try:
        content = json.dumps(state, separators=(',', ':'))
    except Exception:
        logger.exception('Failed to serialize fusion state')
        return
    digest = hashlib.md5(content.encode()).hexdigest()
    if _last_hash.get(path) == digest:
        return
    out_dir = os.path.dirname(path) or '.'
    tmp_fd, tmp_path = tempfile.mkstemp(dir=out_dir, suffix='.tmp')
    try:
        with os.fdopen(tmp_fd, 'w', encoding='utf-8') as f:
            f.write(content)
        shutil.move(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        logger.exception(f'Failed to write fusion state to {path}')
    else:
        _last_hash[path] = digest
        logger.info(f'Wrote fusion state to {path}')
