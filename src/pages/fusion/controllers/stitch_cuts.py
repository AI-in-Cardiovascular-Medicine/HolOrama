"""Intravascular Aligned tab: the tree slices the stitch cuts the end rims and the reconnected
side branches at, picked like an outlet. The stitch itself is in controllers.fusion."""

from __future__ import annotations

import numpy as np
from loguru import logger

from domain.fusion import display_types as colors
from domain.fusion.runtime_types import StitchCuts, StitchCutTarget
from domain.fusion.types import FusionScene
from fusion_pipeline import geometry as geo
from fusion_pipeline import stitch_cuts as cut_ops
from pages.fusion.controllers.base import StageController

SCENE = FusionScene.INTRAVASCULAR_ALIGNED
_RIM_LABELS = {'proximal': 'Proximal rim', 'distal': 'Distal rim'}


class StitchCutController(StageController):
    """Keeps data.stitch_cuts in step with the post-removal snapshot, the tree and the aligned
    geometry it was found on, and lets the user choose a slice per target."""

    @property
    def toolbar(self):
        return self.left_half.alignment_toolbar

    @property
    def viewer(self):
        return self.left_half.viewer

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self.data.stitch_cuts = None
        self._draw()

    def current(self) -> StitchCuts | None:
        """The targets for the current snapshot, tree and alignment, found again when one of
        them changed. Choices stay for targets on the same branch of the same tree."""
        d = self.data
        snapshot, tree = d.results_points_removed, d.vessel_tree
        inputs = (snapshot, tree, d.aligned)
        old = d.stitch_cuts
        if snapshot is None or tree is None or d.aligned is None:
            d.stitch_cuts = None
        elif old is None or any(a is not b for a, b in zip(old.inputs, inputs)):
            try:
                targets = cut_ops.find_targets(geo.primary_geometry(d.aligned), snapshot, tree)
            except Exception as e:
                logger.warning(f'Could not find where the stitch cuts: {e}')
                self.status_bar.showMessage(f'Stitch cuts unknown: {e}')
                d.stitch_cuts = None
            else:
                same_tree = old is not None and old.inputs[1] is inputs[1]
                chosen = cut_ops.keep_choices(old, targets) if same_tree else {}
                d.stitch_cuts = StitchCuts(targets, inputs, chosen)
                if targets:
                    self.viewer.set_layer_visible(SCENE, 'ccta_mesh', True)  # the mesh to pick on
        return d.stitch_cuts

    def refresh(self) -> None:
        self.current()
        self._draw()

    def on_stitched(self, stitched: dict) -> None:
        """Mark the side branches the stitch left loose, with its reason."""
        cuts = self.data.stitch_cuts
        loose = stitched.get('not_reconnected', {})
        if cuts is not None:
            cuts.not_reconnected = dict(loose)
            self._draw()
        for (vessel, branch), why in loose.items():
            logger.warning(f'{self.page.branches.name(vessel, branch)} not reconnected: {why}')
        if loose:
            self.status_bar.showMessage(
                f'Stitched, {len(loose)} side branch(es) left loose: pick another slice on the '
                'Intravascular Aligned tab and stitch again.'
            )

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------

    def select(self, index: int) -> None:
        cuts = self.data.stitch_cuts
        if cuts is not None and 0 <= index < len(cuts.targets):
            cuts.selected = index
            self._draw()

    def on_point_picked(self, x: float, y: float, z: float) -> None:
        """Snap a click to the selected target's slice with the nearest ring point."""
        target = self._target()
        if target is None:
            return
        contours = self._contours(target)
        owners = np.concatenate([np.full(len(c.points_as_tuples()), j) for j, c in enumerate(contours)])
        points = np.vstack([c.points_as_tuples() for c in contours])
        self._choose(target, int(owners[np.argmin(np.linalg.norm(points - np.array([x, y, z]), axis=1))]))

    def on_step(self, step: int) -> None:
        target = self._target()
        if target is None:
            return
        idx = self._slice(target)
        n = len(self._contours(target))
        self._choose(target, min(max((idx if idx is not None else -step) + step, 0), n - 1))

    def on_default(self) -> None:
        target = self._target()
        cuts = self.data.stitch_cuts
        if target is not None and cuts is not None:
            cuts.chosen.pop(target.key, None)
            self._draw()

    def _choose(self, target: StitchCutTarget, idx: int) -> None:
        cuts = self.data.stitch_cuts
        assert cuts is not None
        cuts.chosen[target.key] = idx
        self._draw()

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def _target(self) -> StitchCutTarget | None:
        cuts = self.data.stitch_cuts
        return cuts.targets[cuts.selected] if cuts is not None and cuts.targets else None

    def _contours(self, target: StitchCutTarget) -> list:
        return list(getattr(self.data.vessel_tree, target.vessel)[target.branch].contours)

    def _slice(self, target: StitchCutTarget) -> int | None:
        cuts = self.data.stitch_cuts
        assert cuts is not None
        return cuts.chosen.get(target.key, target.default)

    def _label(self, target: StitchCutTarget) -> str:
        branch = self.page.branches.name(target.vessel, target.branch)
        return f'{_RIM_LABELS[target.key]} ({branch})' if isinstance(target.key, str) else branch

    def _slice_text(self, target: StitchCutTarget) -> str:
        cuts = self.data.stitch_cuts
        assert cuts is not None
        idx = self._slice(target)
        if idx is None:
            return 'no default, pick a slice'
        how = 'chosen' if target.key in cuts.chosen else 'default'
        return f'slice {idx + 1}/{len(self._contours(target))} ({how})'

    def _draw(self) -> None:
        cuts = self.data.stitch_cuts
        target = self._target()
        if cuts is None or target is None:
            for key in ('stitch_cut_slices', 'stitch_cut_selected', 'stitch_cut_others'):
                self.viewer.remove_layer(SCENE, key)
            text = 'No stitch cuts' if cuts is not None else 'Remove labeled points first'
            self.toolbar.set_targets([], -1, text)
            self.left_half.refresh_toolbar(SCENE)
            return

        contours = self._contours(target)
        self.viewer.add_points(
            SCENE,
            'stitch_cut_slices',
            np.vstack([c.points_as_tuples() for c in contours]),
            color=colors.CUT_SLICE_COLORS[target.vessel],
            size=colors.POINT_SIZE,
        )
        idx = self._slice(target)
        if idx is not None:
            ring = np.array(contours[idx].points_as_tuples())
            self.viewer.add_polyline(
                SCENE,
                'stitch_cut_selected',
                np.vstack([ring, ring[:1]]),
                color=colors.CUT_SELECTED_SLICE_COLOR,
                line_width=colors.CUT_SELECTED_SLICE_LINE_WIDTH,
                on_top=True,
            )
        else:
            self.viewer.remove_layer(SCENE, 'stitch_cut_selected')
        # Every other target's slice, to see all cuts at once.
        others = [
            np.array(self._contours(t)[i].points_as_tuples())
            for t in cuts.targets
            if t is not target and (i := self._slice(t)) is not None
        ]
        if others:
            self.viewer.add_points(
                SCENE, 'stitch_cut_others', np.vstack(others), color=colors.CUT_CAP_COLOR, size=colors.POINT_SIZE
            )
        else:
            self.viewer.remove_layer(SCENE, 'stitch_cut_others')

        entries = [(f'{self._label(t)} · {self._slice_text(t)}', cuts.not_reconnected.get(t.key)) for t in cuts.targets]
        self.toolbar.set_targets(entries, cuts.selected, f'{self._label(target)} · {self._slice_text(target)}')
        self.left_half.refresh_toolbar(SCENE)
