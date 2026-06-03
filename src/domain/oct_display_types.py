import numpy as np

# Sepia/copper false-colour LUT matching clinical IVOCT viewers (Abbott OPTIS-style).
# Anchor positions are shifted earlier so yellow/light tones appear at lower pixel
# values (slightly brighter overall). Shape (256, 3) uint8.
_OCT_ANCHORS = np.array(
    [
        [0.00, 0, 0, 0],  # black
        [0.14, 52, 17, 6],  # dark maroon
        [0.32, 135, 56, 18],  # orange-brown
        [0.52, 208, 108, 40],  # full orange
        [0.71, 240, 170, 84],  # orange-tan
        [0.86, 251, 218, 140],  # warm yellow-cream
        [1.00, 255, 242, 190],  # pale warm yellow (highlights)
    ]
)

_x = np.arange(256) / 255.0
OCT_LUT: np.ndarray = np.column_stack(
    [np.interp(_x, _OCT_ANCHORS[:, 0], _OCT_ANCHORS[:, c]) for c in (1, 2, 3)]
).astype(np.uint8)
