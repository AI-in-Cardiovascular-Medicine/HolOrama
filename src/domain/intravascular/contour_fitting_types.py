"""Tuning of the contours the software fits rather than the user draws: read off a mask
(input_output.input.mask_contours) or resampled to another knot count
(domain.intravascular.knot_resampling)."""

# -- reading a mask --------------------------------------------------------------------------

MIN_COMPONENT_PX: int = 20  # anything smaller is noise
OPEN_FRACTION: float = 0.85  # share of its directions a region must reach its container's boundary in to be an arc
TOUCH_TOLERANCE_PX: float = 2.0  # how close to the container's boundary counts as reaching it
RING_GAP_DEG: float = 10.0  # a region whose widest direction gap is narrower is a ring
MAX_MERGE_GAP_DEG: float = 90.0  # widest hidden gap two pieces of one arc are bridged across

# How far a closed contour's spline may stray from (most of) its boundary: half a pixel
# around something small, up to a pixel along a long outline (FIT_TOLERANCE_PER_PX of its length).
FIT_TOLERANCE_PX: tuple[float, float] = (0.5, 1.0)
FIT_TOLERANCE_PER_PX: float = 1 / 300
FIT_SAMPLES: int = 500  # points along a fitted spline it is compared with its boundary at
# (bend weight, phase) of the knot placements tried for a closed contour, plain even spacing first
KNOT_PLACEMENTS: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (0.0, 0.5),
    (1.0, 0.0),
    (1.0, 0.5),
    (3.0, 0.0),
    (3.0, 0.5),
)

# The fibrous cap OCT segmentations label separately (tissue between the lumen and a lipid
# pool) is vessel wall, so it is read as the EEM. The lipid's luminal edge then runs along
# the cap's far side as a real boundary, not one hidden under the cap. Only while the preset
# has a lipid type and no type of its own for this label.
FIBROUS_CAP_LABEL: int = 6
LIPID_ID: str = 'lipid'

# -- resampling to another knot count --------------------------------------------------------

REFERENCE_SAMPLES: int = 400  # points along the original shape a resampled one is compared at
FLAT_PX: float = 0.25  # a spline this close to the original shape everywhere already follows it
PIN_PX: float = 1.0  # how close a start/end label has to sit to a knot to pin it
