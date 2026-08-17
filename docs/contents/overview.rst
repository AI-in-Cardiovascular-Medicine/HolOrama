.. docs/contents/overview.rst

Overview
========

What HolOrama is for
--------------------

HolOrama turns cardiac images into **geometry you can measure, export and reuse**:

- **Annotation.** Draw contours on intravascular (IVUS/OCT) frames or paint labels on a
  CCTA volume, with tools designed to record *how certain* you are about a border rather
  than forcing a single crisp line.
- **Export for machine learning.** Every intravascular annotation can be written out as
  paired image/mask NIfTI files, so a segmentation model can be trained directly on your
  labels.
- **Reconstruction.** Build a 3-D model of the aortic root with the coronaries from CCTA,
  and fuse it with an intravascular pullback into a single geometry — for example as
  input to a computational fluid dynamics study.

The three modules
-----------------

The vertical bar on the far left of the main window switches between the three modules.
They share one window, one menu bar and one status bar, and each keeps its own loaded
data, so you can move back and forth without reloading.

.. rubric:: Intravascular (IVUS / OCT)

Frame-by-frame contouring of a pullback.

- Contour types: ``lumen``, ``EEM``, ``calcium``, ``side branch``, ``lipid``,
  ``macrophage`` — plus distance measurements, a reference point and a wire-shadow angle.
- Every contour is a **closed spline**, an **open spline**, or a closed spline carrying an
  **uncertain region** delimited by a start and end point. Uncertainty is part of the
  annotation, not a comment in a spreadsheet.
- Optional automatic lumen segmentation (source install only, see
  :doc:`installation`).
- Export: report (metrics per frame), contour CSVs, and image/mask **NIfTI** volumes.

IVUS pullbacks additionally get two signal-processing tools:

- :doc:`Image-based gating <modules/gating>` — identifies diastolic and systolic frames
  from the images themselves, no ECG required (the method published as *AIVUS-CAA*).
- :doc:`Breathing-motion detection <modules/breathing>` — extracts the breathing component
  from the lumen-area signal and can reorder the gated frames into a breathing-corrected
  pullback.

.. rubric:: CCTA

Multi-label segmentation and 3-D model building on a CT volume.

- Synchronised axial / coronal / sagittal views of a DICOM folder or NIfTI file.
- **Brush** to add to or erase from any label, on any of the three views.
- **3-D volume rendering** of the visible labels, with a **lasso** tool to delete
  everything of one label inside a drawn region — the fast way to remove noise and
  structures the segmentation picked up by mistake.
- Draw **cut planes** (LVOT and aorta top) and extract the aortic root together with the
  coronaries as one combined NIfTI mask or STL mesh, then **smooth**, **reduce** and — if
  you have `vmtk <https://vmtk.github.io>`_ installed — compute **centerlines**.

.. rubric:: Fusion

A GUI wrapper around `multimodars <https://pypi.org/project/multimodars>`_. It takes what
the other two modules produced and merges them:

- load the cut CCTA geometry and its centerlines, smooth and resample the centerlines,
- label regions on the CCTA (aorta, RCA, LCA, branches),
- align the frames inside the IVUS/OCT pullback, then align that pullback onto the CCTA
  centerline,
- remove the overlapping points, shrink or expand CCTA regions along their centerline so
  the calibre matches the intravascular measurement,
- stitch everything together, remesh and smooth, and export a single STL.

How the modules connect
-----------------------

The modules are usable on their own, but they are designed to feed each other::

    ┌──────────────────────┐        ┌───────────────────────────┐
    │  Intravascular       │        │  CCTA                     │
    │  IVUS / OCT pullback │        │  DICOM folder / NIfTI     │
    └──────────┬───────────┘        └─────────────┬─────────────┘
               │ contour + gate                   │ segment + cut
               ▼                                  ▼
      <case>_csv_files/                  <case>_root_smooth.stl
      diastolic_contours.csv             ao_cl.vtp / rca_cl.vtp / lca_cl.vtp
      systolic_contours.csv                        │
      tagged_contours.csv                          │
               │                                   │
               └───────────────┬───────────────────┘
                               ▼
                     ┌───────────────────┐
                     │  Fusion           │
                     │  → fused STL mesh │
                     └───────────────────┘

The exact file names and where they are written are listed in :doc:`outputs`.

Supported input data
--------------------

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - Module
     - Input
     - Notes
   * - Intravascular
     - DICOM (IVUS/OCT pullback) or NIfTI
     - Opened with **File → Open Intravascular File** (:kbd:`Ctrl+O`). An existing NIfTI
       mask can be loaded on top with **File → Open Intravascular Mask**.
   * - CCTA
     - DICOM folder or NIfTI volume
     - Opened with **File → Open CCTA Folder/File** (:kbd:`Ctrl+Shift+O`). A segmentation
       mask (NIfTI) is loaded with **File → Open CCTA Mask**; without one you start from a
       blank multi-label mask.
   * - Fusion
     - STL/OBJ/PLY mesh, ``.vtp`` centerlines, and a contour-CSV folder
     - Produced by the two modules above, or by any other pipeline that writes the same
       formats.

A first run
-----------

#. Install HolOrama — see :doc:`installation`. The Windows installer is the fastest route.
#. Launch the application. It opens maximised on the **Intravascular** module.
#. Check :doc:`configuration` once — display sizes, colours, auto-save interval and the
   optional vmtk paths all live in ``config.yaml``.
#. Open a case and follow the tutorial for your module:
   :doc:`modules/intravascular`, :doc:`modules/ccta` or :doc:`modules/fusion`.

An example IVUS case ships with the repository under ``test_cases/patient_example`` so you
can follow the intravascular tutorial without your own data.

.. tip::
   Auto-save is on by default (every 10 s, configurable). Contours and tags are written
   next to the file you opened, so re-opening a case restores your work.
