"""Where the stitch cuts: with the vessel tree, multimodars cuts the end rims of the opening
(the proximal one unless it is the ostium) and every side branch it reconnects flat at a tree
slice. These are the targets to choose those slices for, and the stitch arguments they become.
"""

from __future__ import annotations

from typing import Any

from domain.fusion.runtime_types import StitchCuts, StitchCutTarget
from fusion_pipeline import multimodars_api as mm_api

ENDS = ('proximal', 'distal')


def find_targets(iv_geometry, snapshot: dict, tree: Any) -> list[StitchCutTarget]:
    """The rims, proximal first, then the side branches, each with its default slice."""
    found = mm_api.run_stitch_cut_targets(iv_geometry, snapshot, tree)
    targets = [StitchCutTarget(end, *found[end]) for end in ENDS if found[end] is not None]
    # A side branch is cut at its first slice unless chosen otherwise.
    targets += [StitchCutTarget((vessel, branch), vessel, branch, 0) for vessel, branch in found['side_branches']]
    return targets


def stitch_kwargs(cuts: StitchCuts | None) -> dict:
    """run_stitch arguments for the chosen slices. Nothing chosen leaves multimodars' defaults."""
    if cuts is None:
        return {}
    chosen = cuts.chosen
    return {
        'proximal_cut_slice': chosen.get('proximal'),
        'distal_cut_slice': chosen.get('distal'),
        'branch_cut_slices': {key: idx for key, idx in chosen.items() if isinstance(key, tuple)},
    }


def keep_choices(old: StitchCuts | None, targets: list[StitchCutTarget]) -> dict:
    """The slices chosen for `old` targets that are still targets on the same branch."""
    if old is None:
        return {}
    branch_of = {t.key: (t.vessel, t.branch) for t in targets}
    old_branch = {t.key: (t.vessel, t.branch) for t in old.targets}
    return {key: idx for key, idx in old.chosen.items() if key in branch_of and branch_of[key] == old_branch[key]}


def saved_choices(cuts: StitchCuts | None) -> list[dict]:
    """The chosen slices as the case state saves them, with the branch each target is on."""
    if cuts is None:
        return []
    branch_of = {t.key: (t.vessel, t.branch) for t in cuts.targets}
    return [
        {
            'key': list(key) if isinstance(key, tuple) else key,
            'vessel': branch_of[key][0],
            'branch': branch_of[key][1],
            'slice': idx,
        }
        for key, idx in cuts.chosen.items()
        if key in branch_of
    ]


def choices_from_saved(saved: list[dict], targets: list[StitchCutTarget]) -> dict:
    """The saved slices for `targets` still on the same branch (see keep_choices)."""
    old_targets, chosen = [], {}
    for entry in saved:
        raw = entry['key']
        key = (raw[0], int(raw[1])) if isinstance(raw, list) else raw
        old_targets.append(StitchCutTarget(key, entry['vessel'], int(entry['branch']), None))
        chosen[key] = int(entry['slice'])
    return keep_choices(StitchCuts(old_targets, (None, None, None), chosen), targets)
