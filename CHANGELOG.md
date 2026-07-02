# Changelog

All notable changes to this project will be documented in this file.
This project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-07-02

Project renamed from AIVUS-CAA to HolOrama; version numbering restarted.

### Changed
- Project renamed from AIVUS-CAA to HolOrama; `__version__` reset to 0.1.0. Entries below this point are pre-rename history under the AIVUS-CAA name
- Contour JSON files now tagged with an `ho_` marker (e.g. `_contours_ho_0_1_0.json`) so a freshly-saved HolOrama file is always loaded in preference to a pre-rename AIVUS-CAA file, regardless of the (now lower) numeric version
- Removed legacy flat tuple-of-lists contour JSON format support (`_is_legacy`, `_build_contour_legacy`, `_build_frame_data_legacy`); only the current nested-dict format (produced by `asdict`) is read

### Fixed
- Clicking the intravascular display canvas before an image was loaded raised `AttributeError` on `frame_data_dct.get(...)`; `mousePressEvent` now guards on `main_window.image_displayed` like every other interaction handler

---
*Versions below are pre-rename history under the AIVUS-CAA name.*

## [1.8.0] - 2026-06-26

Gating module rewrite, 3D crosshair and lasso erase for the CCTA viewer, and longitudinal view phase markers.

### Added
- Automatic heart-rate detection via FFT peak of the image signal; bandpass filter centered on detected `f_heart`
- Frequency sweep window: heatmap of the image signal at sweeping BPM cutoffs; click a row to apply that cutoff interactively
- Lumen area incorporated as a second gating signal when contours are available
- CCTA 3D crosshair: clicking in the 3D render propagates the voxel position to the 2D views and vice versa
- CCTA lasso erase tool: draw a screen-space polygon to zero mask voxels inside it for the selected label
- Longitudinal view phase markers: dotted cosmetic-pen lines at diastolic and systolic frame positions in their respective phase colors

### Changed
- Peak detection replaced: `scipy.find_peaks` replaced by `walk_extrema` (hysteresis direction-change walker) and `filter_by_period` (drops peaks outside the expected cardiac period by more than 40%)
- Gating frames now selected at image-signal valleys (minimum NCC = stable end-phases) instead of peaks
- Area-signal classification corrected for aortic IVUS: area maximum = systole, area minimum = diastole
- Module renamed from `signal_processing` to `gating_pipeline` and from `contour_based_gating` to `gating_plot`; classes and attributes updated accordingly
- STL export defaults to ASCII format
- 3D viewer click handling moved from VTK observer to Qt eventFilter

### Fixed
- Gating plot freeze on re-run when line artists from a cleared figure could not be removed
- Frequency sweep opened a non-interactive Agg figure; now wrapped in FigureCanvasQTAgg and QMainWindow
- DICOM modality aliases US and OPT added so standard IVUS and OCT files are no longer rejected at load
- DICOM private tag values in the metadata inspector truncated to prevent overflow
- Camera position no longer shifts after lasso erase in the CCTA 3D viewer
- Hidden labels preserved in _actors after lasso erase so they can be re-shown without a full re-render
- Lasso execute is a no-op when no mask is loaded
- RuntimeError from a Qt-deleted Marker object handled when removing the longitudinal view frame marker

## [1.7.0] - 2026-06-09

Aortic root STL extraction with cut-plane workflow and QoL improvements for the CCTA module.

### Added
- Cut-line overlays rendered as dashed grey lines on the 2D CCTA views after a cut plane is drawn
- Progress bars for NIfTI and STL export; file dialog now opens before computation starts so the UI is responsive
- Escape key on CCTA page returns to neutral state: deactivates brush and cancels any active line draw
- Versioned mask auto-loaded on volume open if a matching `_ccta_seg_*.nii.gz` file exists next to the source; falls back to the manual load dialog if none is found

### Changed
- Binary STL write vectorised using a numpy structured dtype (50 bytes per triangle in one `tobytes()` call)

### Fixed
- STL export vertex coordinates now replicate the VTK display coordinate system (Y-flip only), correcting coronary LR orientation and aortic root upright alignment
- STL face winding flipped to produce outward-facing normals (skimage marching cubes emits inward normals by default)

## [1.6.3] - 2026-06-09

### Added
- Autosave functionality for CCTA images (same naming convention as for JSON but as nifti mask)
- Both intravascular and CCTA do saving on thread that copied data, ensuring no interruption of main thread

## [1.6.2] - 2026-06-08

### Added
- Second toolbar row on the intravascular page: contour-type dropdown (Lumen / EEM / Calcium / Branch / Lipid / Macrophage), **New Contour** button, and **+ Add Contour** button
- Drawing-tool buttons (Closed Spline, Open Spline, Brush) are automatically greyed out for tools not permitted by the selected contour type (`ALLOWED_TOOLS`)
- Keyboard shortcuts (E, Q, 7-0, Ctrl+7-0) now also sync the contour-type dropdown
- NIfTI files are now saved next to the opened input file (in a `contoured_frames/`, `gated_frames/`, or `all_frames/` subdirectory) instead of the `nifti_dir` path from `config.yaml`
- `QWidgetWindow must be a top level window` Qt warning on page reload suppressed by calling `hide()` before `removeWidget`
- `RuntimeError: wrapped C/C++ object of type QCheckBox has been deleted` on page reload suppressed with `sip.isdeleted()` guard in `change_value`

## [1.6.1] - 2026-06-08

### Changed
- Remove hydra dependency, because overkill

## [1.6.0] - 2026-06-07

Brush tool for manual mask editing, uv migration, and gating plot interactivity fix.

### Added
- Brush tool for manual mask painting on the intravascular page; radius and color configurable

### Changed
- Switched dependency manager from Poetry to uv and `pyproject.toml` converted to new format

### Fixed
- Gating plot click-to-add and drag-to-move lines broken in PyQt6 due to `cursor().shape() != 0` always evaluating `True` (strict enum vs. bare int) is now replaced with `toolbar.mode` check
- Missing segmentation model file no longer silently swallowed -> shows an error dialog pointing to `config.yaml`

## [1.5.0] - 2026-06-03

Initial CCTA module with orthogonal viewer, 3D mesh renderer, and mask overlay.

### Added
- CCTA tri-plane viewer (axial / coronal / sagittal) with aspect-corrected slicing, synchronised crosshair, window/level, and zoom
- VTK-based 3D surface renderer for segmentation masks with per-label colouring; camera resets to fit geometry on each render
- Mask overlay panel with per-label visibility toggles, editable names, colour swatches, and opacity slider
- CCTA I/O: DICOM-series reader with HU calibration; NIfTI volume and mask reader; BIDS JSON sidecar support
- Sidebar navigation to switch between Intravascular and CCTA pages
- `domain/ccta_display_types.py` and `domain/oct_display_types.py` for display constants and the OCT false-colour LUT
- `WarningMessage` popup class
- Error dialogs when clicking Render 3D without a mask loaded or with all labels hidden
- Warning when a grayscale OCT NIfTI receives the false-colour LUT automatically
- Auto-prompt to load a mask after opening a NIfTI file (IVUS or CCTA); file picker starts in the image folder
- Open Mask remembers the last image folder; resets on new image load

### Changed
- New image load now fully reinstantiates the page (`Master.reload_intravascular` / `reload_ccta`) instead of per-field reset, guaranteeing clean state
- `dicom_dir.py` -> `ccta_io.py`; `_3DViewerCCTA` -> `CctaViewer3D`; `MaskControlTab` -> `MaskPanel`; `tool_tab/` flattened to `mask_panel.py`
- 3D render button moved out of the OpenGL surface into a Qt layout row to fix painting artefacts

### Fixed
- `CctaRuntimeData` initialised `labels` to a `Field` object instead of `[]` due to `field(default_factory=list)` being called inside a manual `__init__`
- Stale runtime state could silently persist across image loads in both Intravascular and CCTA
- VTK `wglMakeCurrent` errors on CCTA page reload; `Finalize()` now called before widget removal
- Re-rendering after hiding labels still rendered all labels; `_hidden_labels` is now tracked and respected
- "Rendering…" label appeared only after rendering completed instead of immediately on click

## [1.4.0] - 2026-05-28

Major refactor ensuring seperation of concerns and identifying several bugs like this.

### Added
- Progress Dialog for reading in images

### Changed
- Complete structure from:
.
├───gating
├───gui
│   ├───left_half
│   ├───popup_windows
│   ├───right_half
│   └───_utils
├───input_output
├───report
└───segmentation

to: 
.
├───domain
├───gating
├───gui
│   ├───left_half
│   ├───popup_windows
│   ├───right_half
│   └───_utils
├───input_output
│   ├───input
│   └───output
├───segmentation
└───tools
With domain now including all the data structures used by the program and report integrated into output.

### Fixed
- With new structure all bugs with improper cleaning of data when loading new image are resolved, since runtime data is stored in `RuntimeData` class directly.

## [1.3.2] - 2026-05-08

### Added
- Multi-pair start/end labelling for closed splines: double-clicking a knot point on a finalized closed spline shows a QMenu popup ("Mark as Start", "Mark as End", "Remove Label"), allowing any number of start/end pairs to be assigned on the same contour
- Per-pair dotted closure arcs: each matched (start, end) pair renders a solid lesion arc and a complementary dotted closure arc with no visual overlap; additional pairs are composited via a boolean point mask so the solid and dotted paths are always strictly non-overlapping

### Changed
- Open splines now auto-assign start/end coordinates to the first and last knot point on finalization; no user interaction required
- `Contour.start_coords` / `end_coords` schema changed from a single `(x, y)` tuple per contour index to a `List[Tuple[float, float]]`; old JSON files are transparently migrated on load via `_normalize_coord_entry`

### Fixed
- Open spline double-click correctly ends the contour at the double-click position (spurious preceding `mousePressEvent` knot intentionally kept in geometry so `full_contour[-1]` snaps to the click location)

## [1.3.1] - 2026-05-06

### Added
- NIfTI image support: `parse_nifti` and `parse_nifti_oct` functions in `metadata.py` mirror the existing DICOM parsers, extracting resolution, pullback speed, pullback length, frame rate, and image dimensions from the NIfTI header via SimpleITK
- NIfTI OCT colour display: RGB frames stored as `images_rgb` so the display renderer and longitudinal view serve colour images, matching the DICOM OCT behaviour

### Fixed
- NIfTI OCT axis ordering: SimpleITK returns scalar 4D NIfTI with channels at dim 0 `(3, F, H, W)`; transposed to channels-last `(F, H, W, 3)` before grayscale conversion
- Bare `except:` in NIfTI loading path replaced with `except Exception` and `traceback.print_exc()` so errors are no longer silently swallowed
- `display.py` and `longitudinal_view.py` now check for `images_rgb` before falling back to `dicom.pixel_array`, preventing an `AttributeError` crash when loading NIfTI OCT files
- `reset_state` clears `images_rgb` to prevent stale NIfTI RGB data bleeding into subsequent DICOM loads

## [1.3.0] - 2026-03-17
Beta version for AIVUS-OCT

### Added
- Panning mode with Ctrl+LMB drag
- Tag every x mm functionality
- Log file clean up function
- Display areas in longitudinal view as dots
- Direct links to report a problem or request a feature under Help menu

### Fixed
- Panning mouse behaviour and Escape key to exit drawing mode
- Cursor shape only shown inside display area
- Bug where slider disappears for lumen and lumen type is always selected after closing contour
- Deleting of end coordinates
- Hardcoded Abbott exception for missing framerate metadata in hospital DICOM headers
- Open spline downsample
- Zooming fix
- Logger warnings
- Guard against garbage collector of other systems

### Changed
- Mouse wheel now scrolls through frames; LMB used for zooming
- Split right-half display logic into separate OCT and IVUS builds
- Fixed monkey patching of right and left half

## [1.2.2] - 2026-03-06
### Added
- Added a logger functionality to main loop
- Added a startup banner

## [1.2.1] - 2026-03-05
Now runs on PyQt6

### Added
- Set up for Windows using GPU acceleration
- A segmentation clean up function, that adds a contour around the catheter if segmentation could not produce one.

### Changed
- updated tests to PyQt6 (not working yet)
- updated README.md

## [1.2.0] - 2026-03-05
Now runs on PyQt6

### Added
- Support for reading OCT data in addition to IVUS.
- Angle measurement tool for quantifying wire shadow artifacts.
- Contour types for EEM, calcium, lipid, macrophage, and side branches.
- Support for drawing single or multiple contours per type (all types except EEM and lumen support multiple).
- Open and closed spline modes; closed splines cast a ray to the EEM boundary.
- Uncertain region annotation: a start and end point can be set to mark the confident region, with the uncertain portion rendered as a dotted line.

### Changed
- Completely refactored the Display module for maintainability and correctness: enforces type safety, proper error handling, and single-responsibility design throughout.
- Separated spline object logic into two distinct classes - one for geometric calculations and one for the PyQt representation.

## [1.1.1] - 2025-10-19
### Changed
- update version in ``__version__.py`` and fix relative paths in ``conf.py``

## [1.1.0] - 2025-09-27
### Added
- Support for segmenting external elastic membrane (EEM), calcium, and branches.
- Refactored contour handling into an `Enum`, ensuring scalability for additional contours.

### Changed
- Adjusted all outputs to align with new contours.

## [1.0.0] - 2025-09-18
### Added
- Documentation published on ReadTheDocs.
- Direct link to the published paper added as a GitHub ribbon.

### Changed
- Declared first stable release (after paper publication).
- Updated citation from medRxiv to *Computer Methods and Programs in Biomedicine*.

[1.8.0]: https://github.com/AI-in-Cardiovascular-Medicine/AIVUS-CAA/compare/v1.7.0...v1.8.0
[1.7.0]: https://github.com/AI-in-Cardiovascular-Medicine/AIVUS-CAA/compare/v1.6.3...v1.7.0
[1.6.0]: https://github.com/AI-in-Cardiovascular-Medicine/AIVUS-CAA/compare/v1.5.0...v1.6.0
[1.5.0]: https://github.com/AI-in-Cardiovascular-Medicine/AIVUS-CAA/compare/v1.4.0...v1.5.0
[1.3.1]: https://github.com/AI-in-Cardiovascular-Medicine/AIVUS-CAA/compare/v1.3.0...v1.3.1
[1.3.0]: https://github.com/AI-in-Cardiovascular-Medicine/AIVUS-CAA/compare/v1.2.1...v1.3.0
[1.2.1]: https://github.com/AI-in-Cardiovascular-Medicine/AIVUS-CAA/compare/v1.2.0...v1.2.1
[1.2.0]: https://github.com/AI-in-Cardiovascular-Medicine/AIVUS-CAA/compare/v1.1.1...v1.2.0
[1.1.1]: https://github.com/AI-in-Cardiovascular-Medicine/AIVUS-CAA/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/AI-in-Cardiovascular-Medicine/AIVUS-CAA/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/AI-in-Cardiovascular-Medicine/AIVUS-CAA/releases/tag/v1.0.0