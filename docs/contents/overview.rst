.. docs/contents/overview.rst

Overview
========

What HolOrama is for
--------------------

HolOrama grew out of a personal need during my PhD. Intravascular ultrasound (IVUS) was
central to uncovering the pathophysiology of coronary artery anomalies, but that meant
segmenting thousands of images and the existing tools felt slow and not flexible enough 
to build an efficient workflow around. So the first goal was simply the most intuitive 
and efficient segmentation tool I could get, fast enough to keep training deep-learning 
models on real data.

The next need had no tool at all: turning intravascular images into 3D models and merging
them with coronary computed tomography angiography (CCTA) into one geometry. That brings in
engineering concepts most physicians never meet — gating, breathing correction, alignment,
fusion — so the challenge became hiding all of it behind an interface any clinician can 
intuitively use.

Hence the two promises of HolOrama: intuitive even where the topic is complex, and flexible
enough to fit your own workflow.

Specifically HolOrama spans the following functionalities:

- **Intravascular segmentation.** Draw contours on IVUS/OCT frames — lumen, EEM, calcium,
  side branches and more — with tools designed to record *how certain* you are about a
  border rather than forcing a single crisp line. On a source install, lumen contours can
  be pre-segmented automatically and then corrected by hand.
- **Motion correction.** IVUS pullbacks carry substantial motion artefacts, from the
  heartbeat and from breathing. HolOrama gates the pullback from the images alone (no ECG
  needed) and can detect and remove the breathing component.
- **Clinical reports.** Per-frame metrics for both IVUS and OCT pullbacks as a report file 
  plus contour CSVs including lumen and EEM area, circumference, longest and shortest diameter, 
  elliptic ratio, pullback position.
- **Export for machine learning.** Every intravascular annotation can be written out as
  paired image/mask NIfTI files, so a segmentation model can be trained directly on your
  labels.
- **CCTA segmentation and 3D geometry.** Paint and correct multi-label masks on a CT
  volume, inspect and clean them in a 3D rendering, cut out the aortic root with the
  coronaries, and compute centerlines with `vmtk <https://vmtk.github.io>`_.
- **Fusion.** A complete GUI wrapper around
  `multimodars <https://pypi.org/project/multimodars>`_ that fuses the CCTA geometry with
  an intravascular pullback into a single mesh — for example as input to a computational
  fluid dynamics study.

The three modules
-----------------

The vertical bar on the far left of the main window switches between the three modules.
They share one window, one menu bar and one status bar, and each keeps its own loaded
data, so you can move back and forth without reloading.

.. rubric:: Intravascular (IVUS / OCT)

Frame-by-frame contouring of a pullback.

- Contour types: ``lumen``, ``EEM``, ``calcium``, ``side branch``, ``lipid``,
  ``macrophage``, ``wire`` (as a shadow angle, several per frame) — plus distance
  measurements and a reference point.
- Every contour is a **closed spline**, an **open spline**, or a closed spline carrying an
  **uncertain region** delimited by a start and end point. Uncertainty is part of the
  annotation, not a comment in a spreadsheet.
- Optional automatic lumen segmentation (source install only, see
  :doc:`installation`).
- Export: report (metrics per frame), contour CSVs, and image/mask **NIfTI** volumes.

.. image:: https://raw.githubusercontent.com/AI-in-Cardiovascular-Medicine/HolOrama/main/media/OCT_demo.gif
   :alt: Contouring an OCT pullback in HolOrama
   :align: center
   :width: 760px

IVUS pullbacks additionally get two signal-processing tools:

- :doc:`Image-based gating <modules/gating>` — identifies diastolic and systolic frames
  from the images themselves, no ECG required (the method published as *AIVUS-CAA*).
- :doc:`Breathing-motion detection <modules/breathing>` — extracts the breathing component
  from the lumen-area signal and can reorder the gated frames into a breathing-corrected
  pullback.

.. image:: https://raw.githubusercontent.com/AI-in-Cardiovascular-Medicine/HolOrama/main/media/IVUS_demo.gif
   :alt: Contouring an IVUS pullback in HolOrama
   :align: center
   :width: 760px

.. rubric:: CCTA

Multi-label segmentation and 3D model building on a CT volume.

- Synchronised axial / coronal / sagittal views of a DICOM folder or NIfTI file.
- **Brush** to add to or erase from any label, on any of the three views.
- **3D volume rendering** of the visible labels, with a **lasso** tool to delete
  everything of one label inside a drawn region — the fast way to remove noise and
  structures the segmentation picked up by mistake.
- Draw **cut planes** (LVOT and aorta top) and extract the aortic root together with the
  coronaries as one combined NIfTI mask or STL mesh, then **smooth**, **reduce** and — if
  you have `vmtk <https://vmtk.github.io>`_ installed — compute **centerlines**.

.. image:: https://raw.githubusercontent.com/AI-in-Cardiovascular-Medicine/HolOrama/main/media/CCTA_demo.gif
   :alt: Segmenting and cleaning a CCTA volume in HolOrama
   :align: center
   :width: 760px

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

.. image:: https://raw.githubusercontent.com/AI-in-Cardiovascular-Medicine/HolOrama/main/media/Fusion_demo.gif
   :alt: Fusing CCTA and intravascular data in HolOrama
   :align: center
   :width: 760px

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

#. Install HolOrama (see :doc:`installation`). With the windows installer no coding knowledge needed.
#. Launch the application. It opens maximised on the **Intravascular** module.
#. Optionally check :doc:`configuration` and adjust display sizes, colours, auto-save interval and the
   optional vmtk paths.
#. Open a case and follow the tutorial for your module:
   :doc:`modules/intravascular`, :doc:`modules/ccta` or :doc:`modules/fusion`.

An example IVUS, OCT and CCTA case are attached to the latest release on Github, so you can follow along.

.. tip::
   Auto-save is on by default (every 10 s, configurable). Saving is additionally performed with every
   contour changing action and can also be triggered with :kbd:`Ctrl+S`. Contours and tags are written
   next to the file you opened, so re-opening a case restores your work.
