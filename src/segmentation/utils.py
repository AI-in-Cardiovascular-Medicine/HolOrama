"""Small helpers taken from nnU-Net.

compute_new_shape is copied unchanged from nnunetv2 2.8.1,
nnunetv2/preprocessing/resampling/default_resampling.py
(https://github.com/MIC-DKFZ/nnUNet, Apache License 2.0,
Copyright 2019 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany).
"""

from typing import List, Tuple, Union

import numpy as np


def compute_new_shape(
    old_shape: Union[Tuple[int, ...], List[int], np.ndarray],
    old_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
    new_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
) -> np.ndarray:
    """Shape of an image of old_shape at old_spacing after resampling to new_spacing.
    From nnunetv2.preprocessing.resampling.default_resampling.compute_new_shape."""
    assert len(old_spacing) == len(old_shape)
    assert len(old_shape) == len(new_spacing)
    new_shape = np.array(
        [int(round(i / j * k)) for i, j, k in zip(old_spacing, new_spacing, old_shape)]
    )
    return new_shape
