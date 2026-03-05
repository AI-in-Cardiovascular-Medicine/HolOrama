# Changelog

All notable changes to this project will be documented in this file.
This project adheres to [Semantic Versioning](https://semver.org/).

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

---

[1.2.0]: https://github.com/AI-in-Cardiovascular-Medicine/AIVUS-CAA/compare/v1.1.1...v1.2.0
[1.1.1]: https://github.com/AI-in-Cardiovascular-Medicine/AIVUS-CAA/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/AI-in-Cardiovascular-Medicine/AIVUS-CAA/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/AI-in-Cardiovascular-Medicine/AIVUS-CAA/releases/tag/v1.0.0