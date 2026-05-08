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
- [x] Subcontours should be deleted seperately
- [x] Can't delete start and end point
- [x] scrolling over end of index hides contours (but only lumen)
- [x] move/center image by mouse drag
- [] clicking in longitudinal view let's user set region of interest
- [] mousewheel in longitudinal view changes cut line

# Lower priority
- [x] Add button to hide measurements
- [x] Make image size dynamic
- [x] Clean up for log files -> only store now when error
- [] Add brush
- [] Reduce memory by not storing main_window.images and main_window.dicom at the same time.
- [] Type safety in whole program
- [] Try/Except in whole program, wherever applicable
- [] Config callable from GUI

# Segmentation tool
- [] Ensure that only tagged frames can be exported
- [x] Adjust output nifti to work with several contour_types
- [] Aim for a first test data set including lumen, side branch, catheter for around 300 frames
- [] Image quality should be tagged
- [x] Number of interactive points should adjust by circumference of the contour -> make half for smaller contours
- [] Find a solution to have several section with start end point
- [] Make mask live update, allow brush mode in mask mode