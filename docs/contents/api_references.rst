.. docs/contents/api_references.rst

Application structure
=====================

This page is for contributors. If you are using HolOrama rather than modifying it, see
:doc:`modules/index` instead.

Source layout
-------------

All application code lives under ``src/``::

    src/
    ├── config.yaml          user-editable settings (see the Configuration page)
    ├── main.py              entry point: creates the QApplication and the Master window
    ├── version.py           __version__ and the contour-file version tag
    ├── domain/              data classes: RuntimeData, FrameData, CctaRuntimeData,
    │                        FusionRuntimeData, contour/tool enums, undo stack
    ├── gui/                 app-level wiring: Master window, page switching, menu bar,
    │                        keyboard shortcuts
    ├── input_output/
    │   ├── input/           readers: DICOM series, NIfTI (IVUS/OCT/CCTA), metadata
    │   └── output/          writers: report, contour CSV/JSON, NIfTI, STL, video
    ├── pages/
    │   ├── intravascular/
    │   │   ├── left_half/       image display, spline contour editor, drawing tools
    │   │   ├── right_half/      gating plot, longitudinal view, phase controls
    │   │   ├── popup_windows/   dialogs: frame range, breathing sort viewer, settings
    │   │   └── utils/           helpers shared across the intravascular page
    │   ├── ccta/
    │   │   ├── left_half/       tri-plane viewer, VTK 3D renderer, cut-geometry viewer
    │   │   ├── right_half/      mask panel, brush panel, STL extraction panel
    │   │   ├── cut_geometry.py  mask → mesh, inlet/outlet location
    │   │   └── vmtk_runner.py   drives an external vmtk install for centerlines
    │   └── fusion/
    │       ├── left_half/       shared 3D viewer and per-scene toolbars
    │       ├── right_half/      the three pipeline columns
    │       └── pipeline.py      thin wrappers around the multimodars calls
    ├── segmentation/        automatic segmentation: nnUZoo wrapper, mask→contour
    ├── signal_processing/   cardiac gating and breathing analysis: signal preparation,
    │                        automatic gating, the interactive gating plot
    └── tools/               Qt-independent helpers: geometry, painting, lasso

Design principles
-----------------

- ``domain/`` is the single source of truth for runtime state. Pages read and write through
  ``RuntimeData`` / ``CctaRuntimeData`` / ``FusionRuntimeData`` rather than keeping their
  own copies.
- ``pages/`` holds all page-specific UI code. Each page (``IntravascularPage``,
  ``CctaPage``, ``FusionPage``) is a self-contained ``QWidget`` instantiated by ``Master``;
  tearing a page down and reinstantiating it (``reload_intravascular`` / ``reload_ccta``)
  is the reset strategy.
- ``tools/`` holds logic reusable across pages with no Qt widget dependency: pure geometry
  and pixmap helpers.
- ``input_output/`` has no GUI imports, so it can be exercised headlessly in tests or CLI
  scripts.
- Long-running work (CCTA volume loading, vmtk centerlines, fusion remeshing) runs in a
  worker thread behind a progress dialog that streams the worker's output, so a slow step
  is distinguishable from a frozen one.

Entry point
-----------

.. code-block:: bash

    python3 src/main.py

``main.py`` creates the ``QApplication``, instantiates ``Master`` (the top-level
``QMainWindow``) and starts the event loop. ``Master`` owns the menu bar, the status bar and
a ``QStackedWidget`` holding the three pages, switched by the vertical navigation bar.

Building these docs
-------------------

.. code-block:: bash

    cd docs && make html

The build is configured in ``docs/conf.py``. Read the Docs builds with
``fail_on_warning: true``, so a Sphinx warning fails the build: every page must be
reachable from a toctree and every cross-reference must resolve.

Per-module API documentation can be generated with ``sphinx.ext.autodoc``, which is already
enabled.
