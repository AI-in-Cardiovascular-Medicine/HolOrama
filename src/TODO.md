# Tools
- [x] open_spline for calcium and macrophage (can draw either one spline then angle to everything behind, or two splines and then between is the mask)
- [] brush when in mask mode, can also use a brush to perfect borders

# Modes
- [x] mask mode showing the actual masks with an alpha value
- [x] showing the current contour on the window

# GUI
- [x] add buttons on top of the image to switch the tool
- [x] mouse scroll should zoom in the image, switching frames sets to default 
(- [] add buttons on the side to choose contour type)

# Bugs
- [] lumen area and circumference have substantially different values when saving to json but not in report
- [x] write report crashes
- [x] When drawing a second contour and the first was closed and the second is open, the double click sets end point to both contours
- [x] Legacy measurements are read in as None