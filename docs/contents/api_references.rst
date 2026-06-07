# docs/contents/api_reference.rst

Application Structure & API Reference
======================================

Source layout
-------------

All application code lives under ``src/``::

    src/
    ├── domain/              data classes — RuntimeData, FrameData, CctaRuntimeData
    ├── gating/              cardiac gating — ContourBasedGating, AutomaticGating, signal processing
    ├── gui/                 app-level wiring — Master window, keyboard shortcuts, menu bar
    ├── input_output/
    │   ├── input/           readers — DICOM series, NIfTI (IVUS/OCT/CCTA), metadata parsers
    │   └── output/          writers — report CSV, NIfTI export, JSON contour save/load
    ├── pages/
    │   ├── ccta/            CCTA page — tri-plane viewer, VTK 3D renderer, mask panel
    │   └── intravascular/
    │       ├── left_half/   image display, spline contour editor, drawing tools
    │       ├── popup_windows/ dialogs — frame range, message boxes, settings
    │       ├── right_half/  gating plot, longitudinal view, phase controls
    │       └── utils/       helpers shared across the intravascular page
    ├── segmentation/        automatic segmentation — nnUZoo wrapper, mask-to-contour conversion
    └── tools/               shared Qt-independent tools — BrushGeometry, BrushCursor

Key design principles
---------------------

- ``domain/`` is the single source of truth for runtime state; pages read and write through ``RuntimeData`` / ``CctaRuntimeData`` rather than storing data locally.
- ``pages/`` contains all page-specific UI code. Each page (``IntravascularPage``, ``CctaPage``) is a self-contained ``QWidget`` instantiated by ``Master`` via ``reload_intravascular`` / ``reload_ccta``; tearing down a page and reinstantiating it is the reset strategy.
- ``tools/`` holds logic that is reusable across pages with no Qt widget dependency (pure geometry and pixmap helpers only).
- ``input_output/`` has no GUI imports; it can be exercised headlessly in tests or CLI scripts.

Entry point
-----------

.. code-block:: bash

    python3 src/main.py

``main.py`` creates the ``QApplication``, instantiates ``Master`` (the top-level ``QMainWindow``), and starts the event loop.

Auto-generated API
------------------

Detailed per-module docs can be generated with ``sphinx.ext.autodoc``::

    cd docs && make html
