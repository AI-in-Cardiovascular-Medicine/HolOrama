.. docs/contents/configuration.rst

Configuration
=============

All settings live in one YAML file. It is worth a look once before your first real case —
display sizes, contour colours, the auto-save interval and the optional vmtk paths are all
set here.

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Installation
     - Location of ``config.yaml``
   * - From source
     - ``src/config.yaml`` in the repository
   * - Windows installer
     - ``%LOCALAPPDATA%\HolOrama\config.yaml``

The packaged application keeps its own writes (logs and config) under ``%LOCALAPPDATA%``
so it runs correctly from read-only install locations such as ``C:\Program Files``. Your
analysis outputs are unaffected and are still written next to the file you opened.

.. tip::
   Most display values can also be changed from inside the running application via
   **Settings → Display Settings…**, which writes them back to ``config.yaml``.

``display``
-----------

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Key
     - Meaning
   * - ``image_size``
     - Side length in pixels of the square box showing the IVUS/OCT image. Default 800.
   * - ``gating_display_stretch``
     - Stretch factor of the gating plot within the right-hand pane.
   * - ``lview_display_stretch``
     - Stretch factor of the longitudinal view within the right-hand pane.
   * - ``windowing_sensitivity``
     - How much level/width changes per pixel dragged with :kbd:`RMB`. Below the default
       is slower, above is faster.
   * - ``zoom_sensitivity``
     - Fraction of zoom applied per pixel dragged. Below 0.005 is slower, above is faster.
   * - ``n_interactive_points``
     - Number of draggable knot points on a new contour. Calcium, lipid, macrophage and
       branch contours default to half of this. Extra points can always be added by
       clicking on the contour line.
   * - ``n_points_contour``
     - Number of points used to represent the interpolated contour outline. Ideally a
       multiple of 100 (used when computing closest points).
   * - ``contour_thickness`` / ``point_thickness`` / ``point_radius``
     - Line and knot-point drawing sizes.
   * - ``color_contour`` / ``color_eem`` / ``color_calcium`` / ``color_branch``
     - Colour per contour type. Accepts any of the 20 predefined Qt colour names or a hex
       code (see `Qt colors <https://doc.qt.io/qt-6/qcolor.html>`_).
   * - ``color_start_point`` / ``color_end_point``
     - Colours of the two markers delimiting an uncertain region (default yellow and red).
   * - ``color_angle``
     - Colour of the wire-shadow angle marker.
   * - ``alpha_contour``
     - Contour fill transparency, 0–255 (higher is more opaque).

``gating``
----------

Parameters of the image-based gating and breathing algorithms — see
:doc:`modules/gating` and :doc:`modules/breathing` for what they do.

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Key
     - Meaning
   * - ``normalize_step``
     - ``0`` computes one global z-score over the whole signal. A value ``> 0`` splits the
       signal into non-overlapping windows of that length and z-scores each separately.
   * - ``f_cardiac_min`` / ``f_cardiac_max``
     - Heart-rate search range in Hz for cardiac-frequency detection. The defaults
       (0.75–3.33 Hz) cover roughly 45–200 bpm, i.e. rest through stress.
   * - ``bandpass_lo_frac``
     - Lower bandpass cutoff as a fraction of the detected cardiac frequency; removes the
       slow pullback trend (sub-cardiac drift).
   * - ``bandpass_hi_frac``
     - Upper bandpass cutoff as a fraction of the detected cardiac frequency; passes the
       2nd harmonic while removing speckle noise.
   * - ``breathing_bins``
     - Number of bins per breathing half-cycle used by the *Filtered* (breathing-corrected)
       sort.

``report``
----------

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Key
     - Meaning
   * - ``plot``
     - Show a plot of the gated-frame results after generating a report.
   * - ``save_as_csv``
     - Also write contour coordinates as CSV files. **Required by the Fusion module** —
       these CSVs are its intravascular input.

``save``
--------

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Key
     - Meaning
   * - ``autosave_interval``
     - Auto-save interval in milliseconds for contours/tags (Intravascular) and the mask
       (CCTA). Default 10000.
   * - ``nifti_dir``
     - Default output directory for images/segmentations exported by the batch script
       ``segment_files.py``.
   * - ``save_niftis``
     - Which frames the batch export writes: ``'contoured'``, ``'all'`` or ``'none'``.
   * - ``save_2d``
     - Also write each frame's image/mask as an individual 2-D NIfTI file.
   * - ``save_3d``
     - Write the full stack of frames as a single 3D NIfTI volume.

``vmtk``
--------

Only used by **Calculate Centerlines** in the CCTA module. vmtk is installed separately by
you — see :ref:`install-vmtk`.

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Key
     - Meaning
   * - ``venv_path``
     - Path to the vmtk Python venv (the directory containing ``bin/activate``).
   * - ``build_path``
     - Path to the vmtk build (the directory containing ``vmtk_env.sh`` and ``bin/``).
   * - ``wsl_distro``
     - Which WSL distribution actually has vmtk's runtime dependencies. Leave empty to use
       whatever ``wsl.exe`` defaults to.

``segmentation``
----------------

Automatic lumen segmentation. Available only in a source install, see
:doc:`installation`.

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Key
     - Meaning
   * - ``model_file``
     - Path to the (nnU-Net) automatic IVUS lumen segmentation model.
   * - ``model_fold``
     - Which model fold to use for inference.
   * - ``normalize``
     - Set to ``True`` when using a TensorFlow model that expects normalised input.
   * - ``input_dir``
     - Input directory used only by the batch script ``segment_files.py``.
   * - ``batch_size``
     - Batch size used during inference.
   * - ``conserve_memory``
     - Set to ``True`` on machines with less than 32 GB RAM. Increases inference time but
       lowers peak memory use.
