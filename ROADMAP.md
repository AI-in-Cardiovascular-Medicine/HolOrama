# Urgent
- [] Reduce memory by not storing main_window.images and main_window.dicom at the same time.
- [x] Currently state pollution, where loading a new image results in rest data from last session, solve by creating a `ImageSession` object which holds the session data.
- [x] Add executable
- [x] Add open spline
- [x] Guard against setting new knot point to close to existing
- [] Adjust gating module to new data

# Lower priority
- [] Add brush
- [x] Add button to hide measurements
- [x] Make image size dynamic