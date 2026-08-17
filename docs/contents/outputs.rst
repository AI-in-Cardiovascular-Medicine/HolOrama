.. docs/contents/outputs.rst

Output files
============

HolOrama writes everything **next to the file you opened** — the case directory is the
project directory. Only logs and ``config.yaml`` live elsewhere (see
:doc:`configuration`).

Throughout this page, ``<case>`` is the opened file without its extension.

Intravascular module
--------------------

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - File
     - Written by / contents
   * - ``<case>_contours_ho_<version>.json``
     - Every contour, tag, phase and the cached gating/breathing state. Written by
       **File → Save** (:kbd:`Ctrl+S`) and by auto-save. Reloaded automatically when the
       case is reopened.
   * - ``<case>_report.txt``
     - Tab-separated per-frame metrics (areas, elliptic ratio, measurements, positions),
       with pullback speed, start frame and frame rate on the first row. Written by
       **File → Save Report** (:kbd:`Ctrl+R`).
   * - ``<case>_csv_files/``
     - Contour coordinates as CSV. Only written when ``report.save_as_csv: True``.
       **This folder is the Fusion module's intravascular input.**
   * - ``<case>_diastolic.npy`` / ``<case>_systolic.npy`` / ``<case>_tagged.npy``
     - Gated (IVUS) or tagged (OCT) frames as stacked image arrays, in acquisition order.
       Written by **File → Save Gated Images**; only the groups that have frames.
   * - ``<case>_pullback.mp4``
     - The pullback as a video, at the acquisition frame rate and the current windowing.
       Written by **File → Save Video Pullback**.

Contents of ``<case>_csv_files/``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All CSVs are **tab-delimited**. One group of files is written per phase family that has
frames — ``diastolic`` and ``systolic`` for gated IVUS, ``tagged`` for OCT.

.. list-table::
   :header-rows: 1
   :widths: 46 54

   * - File
     - Contents
   * - ``<group>_contours.csv``
     - Lumen contour points: ``frame, x_mm, y_mm, position_mm``
   * - ``<group>_reference_points.csv``
     - The reference point per frame, defining the rotational reference
   * - ``eem_<group>_contours.csv``
     - EEM contours, same layout — only when EEM contours exist
   * - ``calcium_<group>_contours.csv``
     - Calcium contours — only when they exist
   * - ``branch_<group>_contours.csv``
     - Side-branch contours — only when they exist
   * - ``combined_sorted_manual.csv``
     - Always written with the report: the gated diastolic frames followed by the systolic
       ones (IVUS) or the tagged frames (OCT), in acquisition order, carrying any manual
       moves made in the breathing-sort viewer.

NIfTI export
~~~~~~~~~~~~

**File → Save NIfTis → Contoured / Gated / All Frames** writes into a subdirectory of the
case directory named after the mode: ``contoured_frames/``, ``gated_frames/`` or
``all_frames/``.

.. list-table::
   :header-rows: 1
   :widths: 46 54

   * - File
     - Written when
   * - ``<case>_img.nii.gz``
     - ``save.save_3d: True`` — the whole selected stack as one 3-D image volume
   * - ``<case>_seg.nii.gz``
     - ``save.save_3d: True`` — the matching 3-D mask volume
   * - ``<case>_frame_<n>_img.nii.gz``
     - ``save.save_2d: True`` — one image file per frame
   * - ``<case>_frame_<n>_seg.nii.gz``
     - ``save.save_2d: True`` — one mask file per contoured frame

Masks are rasterised from the contours through a B-spline interpolation of the knot points
identical to the on-screen spline, with a fixed layering ruleset so overlapping structures
resolve the same way every time. This is the export intended for training a segmentation
model.

CCTA module
-----------

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - File
     - Written by / contents
   * - ``<case>_ccta_seg_<version>.nii.gz``
     - The multi-label mask, at the volume's voxel spacing. Written by **File → Save**
       (:kbd:`Ctrl+S`) and by auto-save. The most recent matching file is auto-loaded when
       the case is reopened.
   * - cut state (alongside the mask)
     - Cut lines, chosen coronaries/aorta/LV labels, RCA and LCA outlet points and custom
       label names. Restored on reopening, and the cut geometry is rebuilt from it.
   * - ``<case>_root_smooth.stl``
     - The cut geometry exactly as fed to vmtk, written by **Calculate Centerlines**.
   * - ``ao_cl.vtp``, ``rca_cl.vtp``, ``lca_cl.vtp``
     - The three computed centerlines. **These are the Fusion module's CCTA input.**
   * - ``ao.csv``, ``rca.csv``, ``lca.csv``
     - The source and target points handed to vmtk, for reproducibility.
   * - chosen path (``.nii.gz`` / ``.stl``)
     - **Extract && Export** result. NIfTI re-derives the combined voxel mask; STL exports
       the built mesh including any smoothing and decimation.

For a DICOM folder, ``<case>`` is the folder path plus the folder's own name; for a NIfTI
volume it is the file without its ``.nii``/``.nii.gz`` extension.

Fusion module
-------------

The fusion module does not auto-save. Its only output is written by
**Export Final Mesh…**: the fused, remeshed and smoothed geometry as a single ``.stl`` at a
path you choose.

Feeding the fusion module
-------------------------

To go from raw data to a fused geometry:

.. list-table::
   :header-rows: 1
   :widths: 26 36 38

   * - Fusion input
     - Produced by
     - File(s)
   * - CCTA Mesh
     - CCTA → **Build Cut Geometry** → **Extract && Export** (STL)
     - your chosen ``.stl``, or ``<case>_root_smooth.stl``
   * - Centerlines
     - CCTA → **Calculate Centerlines**
     - ``ao_cl.vtp``, ``rca_cl.vtp``, ``lca_cl.vtp``
   * - Pullback case folder
     - Intravascular → **File → Save Report** with ``save_as_csv: True``
     - ``<case>_csv_files/``

.. note::
   If ``report.save_as_csv`` is ``False``, no contour CSVs are written and the fusion
   module has nothing to load. Check this before generating the report.

Logs
----

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Installation
     - Log location
   * - From source
     - ``./logs`` in the repository
   * - Windows installer
     - ``%LOCALAPPDATA%\HolOrama``

Attach the relevant log when `reporting a problem
<https://github.com/AI-in-Cardiovascular-Medicine/HolOrama/issues/new?template=bug_report.md>`_.
