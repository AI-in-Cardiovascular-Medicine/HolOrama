.. docs/contents/configuration.rst

Configuration
=============

All settings live in one YAML file. It is worth a look once before your first real case:
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
   * - ``contour_preset``
     - Name of the contour preset to annotate with (see `Contour presets`_ below). The
       ``ccta`` section has its own: the label preset naming and colouring CCTA masks.
   * - ``image_size``
     - Initial side length in pixels of the square box showing the IVUS/OCT image.
       Default 800. Window size adjustable with :kbd:`LMB` drag.
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
       branch contours default to half of this. Contours read from a mask get the full
       number, whatever their type. Extra points can always be added by clicking on the
       contour line, or the count changed with the **Points** box.
   * - ``n_interactive_points_range``
     - Lowest and highest knot count the **Points** box and :kbd:`Shift` + mouse wheel
       allow (default ``[3, 40]``). A contour read from a mask with more knots can still
       be reduced.
   * - ``n_points_contour``
     - Number of points used to represent the interpolated contour outline. Ideally a
       multiple of 100 (used when computing closest points).
   * - ``contour_thickness`` / ``point_thickness`` / ``point_radius``
     - Line and knot-point drawing sizes.
   * - ``insert_point_radius_px``
     - How close to a contour line (in screen pixels) a click must land to insert a knot
       point there (default 20).
   * - ``initial_window_level`` / ``initial_window_width``
     - Brightness (centre) and contrast (width) of the displayed intensity range when an
       image opens and after :kbd:`R` (default 128 and 256).
   * - ``color_start_point`` / ``color_end_point``
     - Colours of the two markers delimiting an uncertain region (default yellow and red).
       Accepts any of the 20 predefined Qt colour names or a hex code (see
       `Qt colors <https://doc.qt.io/qt-6/qcolor.html>`_). Each contour type's own colour
       is set in its contour preset (``src/presets/intravascular``), not here.
   * - ``color_reference``
     - Colour of the reference point (default yellow). Same colour formats as above.
   * - ``angle_handle_radius_mm``
     - How far from the image centre an angular sector's two handles and its arc are
       drawn (default 5 mm). Only the *direction* of a sector's points means anything,
       so this is purely where they are shown. It is pulled inside the image for
       pullbacks whose field of view does not reach that far.
   * - ``alpha_contour``
     - Contour fill transparency, 0–255 (higher is more opaque).

``gating``
----------

Parameters of the image-based gating and breathing algorithms. See
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
       (0.75-3.33 Hz) cover roughly 45-200 bpm, i.e. rest through stress.
   * - ``bandpass_lo_frac``
     - Lower bandpass cutoff as a fraction of the detected cardiac frequency. It removes the
       slow pullback trend (sub-cardiac drift).
   * - ``bandpass_hi_frac``
     - Upper bandpass cutoff as a fraction of the detected cardiac frequency. It passes the
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
     - Also write contour coordinates as CSV files. **Required by the Fusion module**,
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
you (see :ref:`install-vmtk`).

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

Contour presets
---------------

Which contour types the intravascular page offers is set by a **contour preset**, edited in
**Settings → Intravascular Contour Settings…**. Each row is one contour type:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Column
     - Meaning
   * - Label
     - The value the type is written into the exported mask as (1-255).
   * - Colour
     - The colour it is drawn and overlaid in.
   * - Name
     - Its name everywhere in the app.
   * - Tools
     - *Closed only*, *Open and closed*, or *Angle* (a sector about the catheter). The brush
       comes with every spline type.
   * - Inside
     - The type it lies inside. Its region is clipped to that type (and kept out of the
       lumen), and an open contour of it fills outwards from the arc up to that type's
       boundary. Required for open contours.
   * - Layer
     - Where it sits in the mask: wherever two regions overlap, the higher layer shows. A
       type that lies inside another has to be layered above it (the lumen above the EEM, a
       thrombus above the lumen). In the Default preset the lumen is the top layer and the
       wire shadow sits just below it.

The lumen and the EEM are always the first two rows. The keyboard shortcuts go to the rows
in order (``E``, ``Q``, ``7``-``0`` for spline types, ``3`` and ``B`` for angles), and further
rows have none. The built-in **Default** preset is read-only, so duplicate it to change it.

Each preset is one JSON file, so it can be exported and shared. User presets are kept in
``%LOCALAPPDATA%\HolOrama\presets\intravascular`` (Windows installer) or
``presets/intravascular`` in the repository (from source). Every saved contour file also
records the contour types it was drawn with.

The CCTA module has label presets of its own, in ``presets\ccta`` next to them, edited in
**Settings → CCTA Contour Settings…**: each row is a mask value, its colour and its name.
