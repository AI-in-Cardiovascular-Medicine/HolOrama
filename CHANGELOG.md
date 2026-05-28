# Changelog

All notable changes to this project will be documented in this file.
This project adheres to [Semantic Versioning](https://semver.org/).

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
- Separated spline object logic into two distinct classes — one for geometric calculations and one for the PyQt representation.

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

[1.3.1]: https://github.com/AI-in-Cardiovascular-Medicine/AIVUS-CAA/compare/v1.3.0...v1.3.1
[1.3.0]: https://github.com/AI-in-Cardiovascular-Medicine/AIVUS-CAA/compare/v1.2.1...v1.3.0
[1.2.1]: https://github.com/AI-in-Cardiovascular-Medicine/AIVUS-CAA/compare/v1.2.0...v1.2.1
[1.2.0]: https://github.com/AI-in-Cardiovascular-Medicine/AIVUS-CAA/compare/v1.1.1...v1.2.0
[1.1.1]: https://github.com/AI-in-Cardiovascular-Medicine/AIVUS-CAA/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/AI-in-Cardiovascular-Medicine/AIVUS-CAA/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/AI-in-Cardiovascular-Medicine/AIVUS-CAA/releases/tag/v1.0.0