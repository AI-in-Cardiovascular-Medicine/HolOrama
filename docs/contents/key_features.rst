.. docs/contents/key_features.rst

Key Features
============

AIVUS-OCT is designed for IVUS and OCT images in DICOM or NIfTI format and offers the following functionalities:

- **IVUS/OCT Image Inspection:** Frame-by-frame visualization of DICOM or NIfTI images, with display of associated DICOM metadata.
- **Manual Contouring:** Draw one or several contours (lumen, EEM, calcium, side branch, macrophage, lipid) with automatic calculation of several measurements.
  Options include closed spline, open spline, or a closed spline with an uncertain region indicated by start- and end point.
- **Automatic Segmentation:** Automatic segmentation of (currently only IVUS) lumen for all frames.
- **Cardiac Gating:** Automatic gating with extraction of diastolic/systolic frames in IVUS mode; manual tagging of diastolic/systolic frames also supported.
- **Distance Measurements:** Measure up to two distances per frame which will be stored in the report.
- **Wire Shadow:** Indicate the wire shadow using an angle.
- **Mask Generation:** Create automatic masks from contours with predefined rulesets.
- **Session Auto-save:** Automatic saving of contours and tags enabled by default with user-definable interval.
- **Reporting:** Generate a detailed report file containing metrics for each frame.
- **Data Export:** Save coordinate data as CSV files or images and segmentations as NIfTI files (e.g. to train a machine learning model).

