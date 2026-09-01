### What changed

A short description of the change.

### Why

Brief motivation and context. Link the issue this addresses: fixes #<issue-number>

### Which module(s) does this touch?

- [ ] Intravascular (`src/pages/intravascular/`, `src/signal_processing/`)
- [ ] CCTA (`src/pages/ccta/`)
- [ ] Fusion (`src/pages/fusion/`)
- [ ] Shared (`src/domain/`, `src/gui/`, `src/input_output/`, `src/tools/`)
- [ ] Documentation / packaging / CI only

### Checklist

- [ ] Tests pass locally: `pytest`
- [ ] Lint and type checks pass: `pre-commit run --all-files` (black, ruff, mypy — the same checks CI runs)
- [ ] Added or updated tests in `tests/` for the changed behaviour
- [ ] Updated `CHANGELOG.md` under the appropriate heading (Added / Changed / Removed / Fixed)
- [ ] Bumped `__version__` in `src/version.py`, if this warrants a release
- [ ] Updated the documentation in `docs/` if behaviour, shortcuts, configuration keys or output files changed
- [ ] Documentation still builds without warnings: `cd docs && sphinx-build -W -b html . _build/html` (Read the Docs treats warnings as errors)

### Notes for the reviewer

Anything non-obvious: design trade-offs made, parts deliberately left out, or areas you would like a closer look at.

### Verification

How you checked this works — the case or dataset used, which module and steps you exercised, and screenshots for anything visual.
