# Urgent
- [x] Currently state pollution, where loading a new image results in rest data from last session, solve by creating a `ImageSession` object which holds the session data.
- [x] Add executable
- [x] Add open spline
- [x] Guard against setting new knot point to close to existing
- [] Adjust gating module to new data

# Requested features
- [x] Region of interest definition
- [x] Tag every x mm
- [x] Marker on the longitudinal view (to show what has been segmented)
- [x] Marker to show vessel size (outliers)
- [x] Scroll wheel should go through frames
- [x] Frame quality checkboxes
- [] Subcontours should be deleted seperately
- [] clicking in longitudinal view let's user set region of interest
- [] mousewheel in longitudinal view changes cut line

# Lower priority
- [] Reduce memory by not storing main_window.images and main_window.dicom at the same time.
- [] Add brush
- [x] Add button to hide measurements
- [x] Make image size dynamic
- [] Type safety in whole program
- [] Try/Except in whole program, wherever applicable
- [] Clean up for log files
- [] Config callable from GUI