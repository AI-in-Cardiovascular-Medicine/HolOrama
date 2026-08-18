.. docs/contents/modules/index.rst

Modules
=======

HolOrama is split into three modules, switched with the vertical navigation bar on the far
left of the window. Each keeps its own loaded data, so switching does not discard work.

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - I want to…
     - Go to
   * - Draw contours on an IVUS or OCT pullback and export masks for model training
     - :doc:`intravascular`
   * - Find diastolic and systolic frames in an IVUS pullback without an ECG
     - :doc:`gating`
   * - Detect breathing motion and reorder a gated pullback to compensate for it
     - :doc:`breathing`
   * - Segment a CCTA volume, clean it up in 3D and build an aortic-root model
     - :doc:`ccta`
   * - Merge a CCTA model and an intravascular pullback into one geometry
     - :doc:`fusion`

.. toctree::
   :titlesonly:

   intravascular
   gating
   breathing
   ccta
   fusion

Conventions used in these tutorials
-----------------------------------

- **Bold** names refer to on-screen buttons, checkboxes, menu entries and panel titles.
- :kbd:`Keys` are keyboard shortcuts; the full list is in :doc:`../shortcuts`.
- ``Monospace`` refers to file names, paths and configuration keys.
- Every tutorial states its prerequisites at the top. Steps marked *optional* can be
  skipped without breaking the rest of the workflow.

The intravascular tutorials can be followed with the example case shipped in
``test_cases/patient_example``.
