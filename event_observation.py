"""Event measurements in the native sensor grid."""

import numpy as np


def polarity_counts(packet, shape):
    """Count positive and negative events separately, including repeated pixels."""
    x, y, _, p = packet
    height, width = shape
    if np.any((x < 0) | (x >= width) | (y < 0) | (y >= height)):
        raise ValueError('Event coordinates fall outside the native sensor grid')
    pixel = y.astype(np.intp) * width + x.astype(np.intp)
    positive = p > 0
    return np.stack([np.bincount(pixel[sign], minlength=height * width).reshape(shape)
                     for sign in (positive, ~positive)])
